import os
import gc
import json
import random
import argparse
import warnings
from torch_geometric.data import Data
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import pickle
from sklearn.metrics import *
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torch_geometric.nn import global_max_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import *
import torch_scatter
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import Counter
from models.vul_detector import Detector
from helpers import utils
from sklearn.model_selection import train_test_split
from line_extract import get_dep_add_lines_bigvul
# from graph_dataset import VulGraphDataset, collate
from dataset import ContractGraphDataset
from models.gnnexplainer import XGNNExplainer
from models.cfexplainer import CFExplainer
from models.pgexplainer import XPGExplainer, PGExplainer_edges
from models.subgraphx import SubgraphX
from models.gnn_lrp import GNN_LRP
from models.deeplift import DeepLIFT
from models.gradcam import GradCAM

warnings.filterwarnings("ignore", category=UserWarning)


def calculate_metrics(y_true, y_pred):
    results = {
        'binary_precision': round(precision_score(y_true, y_pred, average='binary'), 4),
        'binary_recall': round(recall_score(y_true, y_pred, average='binary'), 4),
        'binary_f1': round(f1_score(y_true, y_pred, average='binary'), 4),
    }
    return results


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def train(args, train_dataloader, valid_dataloader, test_dataloader, model):
    # 设置训练参数
    args.max_steps = args.num_train_epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)

    # 优化器配置
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.max_steps * 0.1,
        num_training_steps=args.max_steps
    )

    # 加载检查点
    checkpoint_last = os.path.join(args.model_checkpoint_dir, 'checkpoint-last')
    # optimizer.load_state_dict(torch.load(optimizer_last, map_location=args.device)strict=False)
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    print(optimizer_last)
    print(scheduler_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    
    # if os.path.exists(scheduler_last):
    #     scheduler.load_state_dict(torch.load(scheduler_last, map_location=args.device), strict=False)
    # if os.path.exists(optimizer_last):
    #     optimizer.load_state_dict(torch.load(optimizer_last, map_location=args.device), strict=False)

    # 打印训练信息
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader.dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Total optimization steps = {args.max_steps}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    # 初始化训练状态
    global_step = args.start_step
    tr_loss = logging_loss = avg_loss = 0.0
    tr_nb = tr_num = train_loss = 0
    best_acc = 0.0

    # 清空梯度
    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training epoch {idx}")
        tr_num = 0
        train_loss = 0
        
        for step, batch_data in enumerate(bar):
            try:
                torch.cuda.empty_cache()
                # 数据完整性检查
                if batch_data is None or not hasattr(batch_data, 'x') or not hasattr(batch_data, 'edge_index'):
                    continue

                # 数据预处理
                batch_data = batch_data.to(args.device)
                x = batch_data.x
                edge_index = batch_data.edge_index.long()
                batch = batch_data.batch

                # 检查数据维度
                num_nodes = x.size(0)
                if edge_index.max() >= num_nodes:
                    print(f"Warning: edge_index max value ({edge_index.max()}) >= num_nodes ({num_nodes})")
                    continue

                # 添加自环边和边的预处理
                edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
                edge_index = coalesce(edge_index)

                # 处理标签
                labels = torch_scatter.segment_csr(batch_data.y, batch_data.ptr).long()
                labels[labels != 0] = 1

                # 训练模式
                model.train()

                # 前向传播
                probs = model(x, edge_index, batch)
                labels = F.one_hot(1 - labels, 2)
                loss = F.binary_cross_entropy(probs, labels.float())

                # 梯度累积
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # 更新统计信息
                tr_loss += loss.item()
                tr_num += 1
                train_loss += loss.item()
                avg_loss = round(train_loss / tr_num, 5)
                bar.set_description(f"epoch {idx} loss {avg_loss}")

                # 优化器步骤
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                    # 计算平均损失
                    if tr_nb == 0:
                        avg_loss = tr_loss
                    else:
                        avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                    # 记录日志
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logging_loss = tr_loss
                        tr_nb = global_step

                    # 保存检查点
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # 验证
                        results = evaluate(args, valid_dataloader, model)
                        print(f"  Valid acc: {results['eval_acc']:.4f}")

                        # 保存最佳模型
                        if results['eval_acc'] > best_acc:
                            best_acc = results['eval_acc']
                            print("  " + "*" * 20)
                            print(f"  Best acc: {best_acc:.4f}")
                            print("  " + "*" * 20)

                            # 保存模型
                            checkpoint_prefix = 'checkpoint-best-acc'
                            output_dir = os.path.join(args.model_checkpoint_dir, checkpoint_prefix)
                            os.makedirs(output_dir, exist_ok=True)
                            
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_path = os.path.join(output_dir, 'model.bin')
                            torch.save(model_to_save.state_dict(), output_path)
                            print(f"Saving model checkpoint to {output_path}")

                            # 测试
                            test_result = evaluate(args, test_dataloader, model)
                            for key, value in test_result.items():
                                print(f"  {key} = {value:.4f}")

                # 清理GPU缓存
                if step % 100 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"Error in batch: {e}")
                print(f"Batch info: nodes={x.size(0)}, edges={edge_index.size(1)}")
                continue

        bar.close()
        
        # 每个epoch结束后保存检查点
        checkpoint_prefix = 'checkpoint-last'
        output_dir = os.path.join(args.model_checkpoint_dir, checkpoint_prefix)
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataloader, model):
    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(eval_dataloader)))
    print("  Batch size = {}".format(args.batch_size))
    args.batch_size=16

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for step, batch_data in enumerate(eval_dataloader):
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
            edge_index = coalesce(edge_index)
            # labels = global_max_pool(batch_data._VULN, batch).long()
            labels = torch_scatter.segment_csr(batch_data.y, batch_data.ptr).long()
            labels[labels != 0] = 1
            probs = model(x, edge_index, batch)
            probs = F.one_hot(torch.argmax(probs, dim=-1), 2)[:, 0]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, 0)
    all_labels = np.concatenate(all_labels, 0)
    eval_acc = np.mean(all_labels == all_probs)

    result = {
        "eval_acc": round(eval_acc, 4),
    }

    eval_results = calculate_metrics(all_labels, all_probs)
    result.update(eval_results)

    return result


def gen_exp_lines(edge_index, edge_weight, index, num_nodes, lines):
    temp = torch.zeros_like(edge_weight).to(edge_index.device)
    temp[index] = edge_weight[index]

    adj_mask = torch.sparse_coo_tensor(edge_index, temp, [num_nodes, num_nodes])
    adj_mask_binary = to_dense_adj(edge_index[:, temp != 0], max_num_nodes=num_nodes).squeeze(0)

    out_degree = torch.sum(adj_mask_binary, dim=1)
    out_degree[out_degree == 0] = 1e-8
    in_degree = torch.sum(adj_mask_binary, dim=0)
    in_degree[in_degree == 0] = 1e-8

    line_importance_init = torch.ones(num_nodes).unsqueeze(-1).to(edge_index.device)
    line_importance_out = torch.spmm(adj_mask, line_importance_init) / out_degree.unsqueeze(-1)
    line_importance_in = torch.spmm(adj_mask.T, line_importance_init) / in_degree.unsqueeze(-1)
    line_importance = line_importance_out + line_importance_in

    ret = sorted(
        list(
            zip(
                line_importance.squeeze(-1).cpu().numpy(),
                lines,
            )
        ),
        reverse=True,
    )

    filtered_ret = []
    for i in ret:
        if i[0] > 0:
            filtered_ret.append(int(i[1]))

    return filtered_ret


def eval_exp(exp_saved_path, model, correct_lines, args):
    graph_exp_list = torch.load(exp_saved_path, map_location=args.device)
    print("Number of explanations:", len(graph_exp_list))

    accuracy = 0
    precisions = []
    recalls = []
    F1s = []
    pn = []
    for graph in graph_exp_list:
        graph.to(args.device)
        x, edge_index, edge_weight, pred, batch = graph.x, graph.edge_index.long(), graph.edge_weight, graph.pred, graph.batch
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        exp_label_lines = correct_lines[int(sampleid)]
        # exp_label_lines = list(exp_label_lines["removed"]) + list(exp_label_lines["depadd"])
        exp_label_lines = list(exp_label_lines["removed"])
        if len(edge_weight) > args.KM:
            value, index = torch.topk(edge_weight, k=args.KM)
        else:
            index = torch.arange(edge_weight.shape[0])
        temp = torch.ones_like(edge_weight)
        temp[index] = 0
        cf_index = temp != 0

        lines = graph._LINE.cpu().numpy()
        exp_lines = gen_exp_lines(edge_index, edge_weight, index, x.shape[0], lines)

        for i, l in enumerate(exp_lines):
            if l in exp_label_lines:
                accuracy += 1
                break

        hit = 0
        for i, l in enumerate(exp_lines):
            if l in exp_label_lines:
                hit += 1
        if hit != 0:
            precision = hit / len(exp_lines)
            recall = hit / len(exp_label_lines)
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            precision = 0
            recall = 0
            f1 = 0
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(f1)

        fac_edge_index = edge_index[:, index]
        fac_edge_index, _ = add_self_loops(fac_edge_index, num_nodes=x.shape[0])  # add self-loop
        fac_logits = model(x, fac_edge_index, batch)
        fac_pred = F.one_hot(torch.argmax(fac_logits, dim=-1), 2)[0][0]

        cf_edge_index = edge_index[:, cf_index]
        cf_edge_index, _ = add_self_loops(cf_edge_index, num_nodes=x.shape[0])  # add self-loop
        cf_logits = model(x, cf_edge_index, batch)
        cf_pred = F.one_hot(torch.argmax(cf_logits, dim=-1), 2)[0][0]

        pn.append(int(cf_pred != pred))

        if args.case_sample_ids and str(sampleid) in args.case_sample_ids:
            case_saving_dir = str(utils.cache_dir() / f"cases")
            case_graph_saving_path = f"{case_saving_dir}/{args.gnn_model}_{args.ipt_method}_{sampleid}.pt"
            torch.save(graph, case_graph_saving_path)
            print(f"Saving {str(sampleid)} in {case_graph_saving_path}!")

    accuracy = round(accuracy / len(graph_exp_list), 4)
    print("Accuracy:", accuracy)
    precision = round(np.mean(precisions), 4)
    print("Precision:", precision)
    recall = round(np.mean(recalls), 4)
    print("Recall:", recall)
    f1 = round(np.mean(F1s), 4)
    print("F1:", f1)
    PN = round(sum(pn) / len(pn), 4)
    print("Probability of Necessity:", PN)

    if args.hyper_para:
        para_saving_dir = str(utils.cache_dir() / f"parameter_analysis")
        if not os.path.exists(para_saving_dir):
            os.makedirs(para_saving_dir)
        if args.ipt_method == "cfexplainer":
            if args.cfexp_L1:
                para_saving_path = os.path.join(para_saving_dir, f"{args.ipt_method}_L1_{args.cfexp_alpha}.res")
            else:
                para_saving_path = os.path.join(para_saving_dir, f"{args.ipt_method}_{args.cfexp_alpha}.res")
        KM_index_map = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6, 16: 7, 18: 8, 20: 9}
        if os.path.isfile(para_saving_path):
            result = json.load(open(para_saving_path, "r"))
        else:
            GNN_models = ["GCNConv", "GatedGraphConv", "GINConv", "GraphConv"]
            metrics = [r"Accuracy", r"Precision", r"Recall", r"$F_1$", r"PN"]
            result = {}
            for GNN_model in GNN_models:
                result[GNN_model] = {}
                for metric in metrics:
                    result[GNN_model][metric] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        result[args.gnn_model][r"Accuracy"][KM_index_map[args.KM]] = accuracy
        result[args.gnn_model][r"Precision"][KM_index_map[args.KM]] = precision
        result[args.gnn_model][r"Recall"][KM_index_map[args.KM]] = recall
        result[args.gnn_model][r"$F_1$"][KM_index_map[args.KM]] = f1
        result[args.gnn_model][r"PN"][KM_index_map[args.KM]] = PN
        json.dump(result, open(para_saving_path, "w"))
    else:
        results_saving_dir = str(utils.cache_dir() / f"results")
        if not os.path.exists(results_saving_dir):
            os.makedirs(results_saving_dir)
        results_saving_path = os.path.join(results_saving_dir, f"{args.ipt_method}.res")
        KM_index_map = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6, 16: 7, 18: 8, 20: 9}
        if os.path.isfile(results_saving_path):
            result = json.load(open(results_saving_path, "r"))
        else:
            GNN_models = ["GCNConv", "GatedGraphConv", "GINConv", "GraphConv"]
            metrics = [r"Accuracy", r"Precision", r"Recall", r"$F_1$", r"PN"]
            result = {}
            for GNN_model in GNN_models:
                result[GNN_model] = {}
                for metric in metrics:
                    result[GNN_model][metric] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        result[args.gnn_model][r"Accuracy"][KM_index_map[args.KM]] = accuracy
        result[args.gnn_model][r"Precision"][KM_index_map[args.KM]] = precision
        result[args.gnn_model][r"Recall"][KM_index_map[args.KM]] = recall
        result[args.gnn_model][r"$F_1$"][KM_index_map[args.KM]] = f1
        result[args.gnn_model][r"PN"][KM_index_map[args.KM]] = PN
        json.dump(result, open(results_saving_path, "w"))


def gnnexplainer_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explainer = XGNNExplainer(
        model=model, explain_graph=True, epochs=800, lr=0.05,
        coff_edge_size=0.001, coff_edge_ent=0.001
    )
    explainer.device = args.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(x, edge_index, False, None,
                                                                                     num_classes=args.num_classes)
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def cfexplainer_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explainer = CFExplainer(
        model=model, explain_graph=True, epochs=800, lr=0.05, alpha=args.cfexp_alpha, L1_dist=args.cfexp_L1
    )
    explainer.device = args.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph.y, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(x, edge_index, False, None,
                                                                                     num_classes=args.num_classes)
        edge_weight = 1 - edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def pgexplainer_run(args, model, eval_model, train_dataset, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    input_dim = args.gnn_hidden_size * 2

    pgexplainer = XPGExplainer(model=model, in_channels=input_dim, device=args.device, explain_graph=True, epochs=100,
                               lr=0.005,
                               coff_size=0.01, coff_ent=5e-4, sample_bias=0.0, t0=5.0, t1=1.0)
    pgexplainer_saving_path = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/pgexplainer.bin")
    if os.path.isfile(pgexplainer_saving_path) and not args.ipt_update:
        print("Load saved PGExplainer model...")
        pgexplainer.load_state_dict(torch.load(pgexplainer_saving_path, map_location=args.device))
    else:
        pgexplainer.train_explanation_network(train_dataset)
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        pgexplainer.load_state_dict(torch.load(pgexplainer_saving_path, map_location=args.device))

    pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer, model=eval_model)
    pgexplainer_edges.device = pgexplainer.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = pgexplainer_edges(x, edge_index,
                                                                                             num_classes=args.num_classes,
                                                                                             sparsity=0.5)
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def subgraphx_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()

    explanation_saving_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/subgraphx")
    if not os.path.exists(explanation_saving_dir):
        os.makedirs(explanation_saving_dir)
    subgraphx = SubgraphX(model, args.num_classes, args.device, explain_graph=True,
                          verbose=False, c_puct=10.0, rollout=5, high2low=False, min_atoms=5, expand_atoms=14,
                          reward_method='gnn_score', subgraph_building_method='zero_filling',
                          save_dir=explanation_saving_dir)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        saved_MCTSInfo_list = None
        prediction = prob.argmax(-1).item()
        if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt')):
            saved_MCTSInfo_list = torch.load(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'),
                                             map_location=args.device)
            print(f"load example {sampleid}.")
        explain_result = subgraphx.explain(x, edge_index, label=prediction, node_idx=0,
                                           saved_MCTSInfo_list=saved_MCTSInfo_list)
        torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'))
        node_weight = torch.zeros(x.shape[0])
        for item in explain_result:
            node_weight[item['coalition']] += item['P']
        node_weight = node_weight / len(explain_result)
        edge_index, _ = remove_self_loops(edge_index.detach().cpu())
        edge_weight = node_weight[edge_index[0]] + node_weight[edge_index[1]]
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def gnn_lrp_run(args, model, test_dataset, correct_lines):
    # for name, parameter in model.named_parameters():
    #     print(name)

    graph_exp_list = []
    visited_sampleids = set()

    explanation_saving_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/gnn_lrp")
    if not os.path.exists(explanation_saving_dir):
        os.makedirs(explanation_saving_dir)
    gnnlrp_explainer = GNN_LRP(model, explain_graph=True)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])

        if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt')):
            edge_masks, self_loop_edge_index = torch.load(
                os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'), map_location=args.device)
            print(f"load example {sampleid}.")
        else:
            walks, edge_masks, related_preds, self_loop_edge_index = gnnlrp_explainer(x, edge_index, sparsity=0.5,
                                                                                      num_classes=args.num_classes)
            torch.save((edge_masks, self_loop_edge_index),
                       os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'))

        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)].sigmoid()
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph.detach().clone().cpu())
        visited_sampleids.add(sampleid)

        del graph
        gc.collect()

    return graph_exp_list


def deeplift_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    deep_lift = DeepLIFT(model, explain_graph=True)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = deep_lift(x, edge_index, sparsity=0.5,
                                                                                     num_classes=args.num_classes)
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)].sigmoid()
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def gradcam_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    gc_explainer = GradCAM(model, explain_graph=True)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(sampleid)

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = gc_explainer(x, edge_index, sparsity=0.5,
                                                                                        num_classes=args.num_classes)
        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def collate(batch):
    """
    自定义批处理函数
    """
    # 获取批次中的最大节点数
    max_nodes = max(data.x.size(0) for data in batch)

    batch_x = []
    batch_edge_index = []
    batch_mask = []
    batch_y = []

    cum_nodes = 0
    for i, data in enumerate(batch):
        num_nodes = data.x.size(0)

        # 填充节点特征到最大节点数
        padded_x = torch.zeros((max_nodes, data.x.size(1)))
        padded_x[:num_nodes] = data.x
        batch_x.append(padded_x)

        # 调整边索引
        if data.edge_index.size(1) > 0:
            edge_index = data.edge_index + cum_nodes
            batch_edge_index.append(edge_index)

        # 创建mask
        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:num_nodes] = True
        batch_mask.append(mask)

        # 收集标签（假设标签存储在_VULN中）
        if hasattr(data, '_VULN'):
            batch_y.append(data._VULN)

        cum_nodes += num_nodes

    # 组合所有批次数据
    batch_x = torch.stack(batch_x)
    batch_edge_index = torch.cat(batch_edge_index, dim=1) if batch_edge_index else torch.empty((2, 0))
    batch_mask = torch.stack(batch_mask)
    if batch_y:
        batch_y = torch.stack(batch_y)

    return Batch(
        x=batch_x,
        edge_index=batch_edge_index,
        mask=batch_mask,
        y=batch_y if batch_y else None,
        batch=torch.arange(len(batch)).repeat_interleave(max_nodes)
    )


def load_datasets(args):
    # 创建数据集
    train_dataset = ContractGraphDataset(
        root_dir=str(args.data_dir),
        partition='train'
    )

    valid_dataset = ContractGraphDataset(
        root_dir=str(args.data_dir),
        partition='val'
    )

    test_dataset = ContractGraphDataset(
        root_dir=str(args.data_dir),
        partition='test'
    )

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate
    )

    return train_dataloader, valid_dataloader, test_dataloader
def label_to_tensor(label):
    label_mapping = {
        'reentrancy': 0,
        'arithmetic': 1,
        'time_manipulation': 2,
    }
    return torch.tensor([label['reentrancy'], label['arithmetic'], label['time_manipulation']], dtype=torch.long)

class FeatureExtractor:
    def __init__(self, bert_path="/root/autodl-tmp/bert-tiny", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path).to(device)
        self.transform = nn.Linear(128, 4).to(device)  # bert-tiny的输出维度是128
        
    def get_features(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        # 将输入移到对应设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]
        
    def get_transformed_features(self, text):
        features = self.get_features(text)
        features = features.squeeze(0)
        return self.transform(features)
    
    def batch_get_features(self, texts, batch_size=32):
        features_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", max_length=512, 
                                  truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bert(**inputs)
                batch_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, 128]
                transformed_features = self.transform(batch_features)  # [batch_size, 4]
                features_list.append(transformed_features)
                
        return torch.cat(features_list, dim=0)
def get_bert_features(text, feature_extractor):
    features = feature_extractor.get_features(text)
    # 将BERT特征转换为4维特征（根据你的Detector输入需求）
    features = features.squeeze(0)  # 移除batch维度
    # 这里需要根据你的具体需求来设计特征转换方式
    # 示例：使用一个简单的线性变换
    transform = nn.Linear(features.shape[-1], 4)
    return transform(features)
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
def load_data(path, all_data_path, model, min_nodes=3):
    if os.path.exists(all_data_path):
        print("Loading preprocessed data from file.")
        all_data = torch.load(all_data_path)
        return all_data

    print("Preprocessed data file not found. Processing data...")
    all_data = []
    feature_extractor = FeatureExtractor()  # 将特征提取器移到循环外
    dataList = load_pkl('/root/autodl-tmp/smartbugs_filenames.pkl')[0:200]

    
    path="/root/autodl-tmp/SCcfg/processed"
    for filename in tqdm(dataList, desc='Loading data'):
        if not filename.endswith('.pt'):
            continue
            
        try:
            file_path = os.path.join(path, filename)
            data = torch.load(file_path)
            
            # 检查数据完整性
            if 'SCcfg' not in data or 'label' not in data:
                print(f"Skipping {filename}: Missing required fields")
                continue

            cfg_graph = data["SCcfg"]["SCcfg"]
            if not cfg_graph or len(cfg_graph) < 2:
                print(f"Skipping {filename}: Invalid cfg_graph structure")
                continue

            # 提取节点和边
            try:
                graph_nodes = list(cfg_graph[0].values())
                subgraph_edges = list(cfg_graph[1].values())
            except (IndexError, AttributeError) as e:
                print(f"Skipping {filename}: Error in graph structure - {str(e)}")
                continue

            # 检查节点数量
            if len(graph_nodes) < min_nodes:
                continue

            # 提取文本并构建特征
            try:
                texts = [node['Expr']['str'] for node in graph_nodes]
                node_features = feature_extractor.batch_get_features(texts)
            except (KeyError, Exception) as e:
                print(f"Skipping {filename}: Error in feature extraction - {str(e)}")
                continue

            # 构建边索引
            if subgraph_edges:
                try:
                    edge_index = torch.tensor([[edge[0], edge[1]] for edge in subgraph_edges], 
                                            dtype=torch.long).t().contiguous()
                except Exception as e:
                    print(f"Skipping {filename}: Error in edge construction - {str(e)}")
                    continue
            else:
                edge_index = torch.zeros([2, 0], dtype=torch.long)

            # 转换标签
            try:
                label = data['label']
                label_tensor = label_to_tensor(label)
            except Exception as e:
                print(f"Skipping {filename}: Error in label conversion - {str(e)}")
                continue

            # 创建图数据对象
            try:
                graph_data = Data(x=node_features, 
                                edge_index=edge_index, 
                                y=label_tensor)
                all_data.append(graph_data)
            except Exception as e:
                print(f"Skipping {filename}: Error in graph construction - {str(e)}")
                continue

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    if all_data:
        try:
            torch.save(all_data, all_data_path)
            print(f"Saved {len(all_data)} processed graphs to {all_data_path}")
        except Exception as e:
            print(f"Warning: Could not save processed data - {str(e)}")
    else:
        print("No valid graphs were processed!")

    return all_data
def balance_classes(data, labels, target_count=700):
    label_to_data = {label: [] for label in set(labels)}
    for item, label in zip(data, labels):
        label_to_data[label].append(item)

    balanced_data = []
    balanced_labels = []

    for label, items in label_to_data.items():
        if len(items) > target_count:
            # 负采样
            sampled_items = random.sample(items, target_count)
        else:
            # 正采样
            sampled_items = items
            sampled_items += random.choices(items, k=target_count - len(items))

        balanced_data.extend(sampled_items)
        balanced_labels.extend([label] * target_count)

    return balanced_data, balanced_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='which gpu to use if any')
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")

    # GNN Model
    parser.add_argument("--model_checkpoint_dir", default="saved_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--gnn_model", default="GCNConv", type=str,
                        help="GNN core.")
    parser.add_argument("--gnn_hidden_size", default=256, type=int,
                        help="hidden size of gnn.")
    parser.add_argument("--gnn_feature_dim_size", default=768, type=int,
                        help="feature dim size of gnn.")
    parser.add_argument("--residual", action='store_true',
                        help="Whether to obtain residual representations.")
    parser.add_argument("--graph_pooling", default="mean", type=str,
                        help="The operator of graph pooling.")
    parser.add_argument("--num_gnn_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--num_ggnn_steps", default=3, type=int,
                        help="The sequence length for GGNN.")
    parser.add_argument("--ggnn_aggr", default="add", type=str,
                        help="The aggregation scheme to use for GGNN.")
    parser.add_argument("--gin_eps", default=0., type=float,
                        help="Eps value for GIN.")
    parser.add_argument("--gin_train_eps", action='store_true',
                        help="If set to True, eps will be a trainable parameter.")
    parser.add_argument("--gconv_aggr", default="mean", type=str,
                        help="The aggregation scheme to use.")
    parser.add_argument("--dropout_rate", default=0.1, type=float,
                        help="Dropout rate.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")

    # Training
    parser.add_argument("--num_train_epochs", default=50, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size.")
    parser.add_argument("--learning_rate", default=5e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_explain", action='store_true',
                        help="Whether to run explaining.")

    # Explainer
    parser.add_argument("--ipt_method", default="gnnexplainer", type=str,
                        help="The save path of interpretations.")
    parser.add_argument("--ipt_update", action='store_true',
                        help="Whether to update interpretations.")
    parser.add_argument("--KM", default=8, type=int,
                        help="The size of explanation subgraph.")
    parser.add_argument("--cfexp_L1", action='store_true',
                        help="Whether to use L1 distance item.")
    parser.add_argument("--cfexp_alpha", default=0.9, type=float,
                        help="CFExplainer.")
    parser.add_argument("--hyper_para", action='store_true',
                        help="Whether to tune the hyper-parameters.")
    parser.add_argument("--case_sample_ids", nargs='+',
                        help="Ids of samples to extract for case study.")

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    print(device)
    args.device = device
    args.model_checkpoint_dir = str(utils.cache_dir() / f"{args.model_checkpoint_dir}" / args.gnn_model)
    print(args.model_checkpoint_dir)
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    model = Detector(args)
    model.to(args.device)
    args.data_dir = "/root/autodl-tmp/SCcfg/processed"
    all_data = load_data(args.data_dir , "",model)
    labels = [data.y[2].item() for data in all_data]
    balanced_data, balanced_labels = balance_classes(all_data, labels)

    train_data, temp_data, train_labels, temp_labels = train_test_split(
    balanced_data, balanced_labels, test_size=0.3, stratify=balanced_labels, random_state=2865)

    valid_data, test_data, valid_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.5, stratify=temp_labels, random_state=2865)
    print("Original data size:", len(all_data))
    print("After balancing:", len(balanced_data))
    print("Train size:", len(train_data))
    print("Valid size:", len(valid_data))
    print("Test size:", len(test_data))

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)
    print("testLoaderSize",len(test_dataloader))

    # train_data, valid_data, train_labels, valid_labels = train_test_split(
    #     all_data, labels, test_size=0.2, stratify=labels, random_state=32)

    # 输出训练和验证集类别分布检查
    train_labels = [data.y[2].item() for data in train_data]
    valid_labels = [data.y[2].item() for data in valid_data]
    test_labels = [data.y[2].item() for data in test_data]
    print(f"训练集类别分布: {dict(Counter(train_labels))}")
    print(f"验证集类别分布: {dict(Counter(valid_labels))}")
    print(f"测试集类别分布: {dict(Counter(test_labels))}")

    # train_counter = Counter(train_labels)
    # valid_counter = Counter(valid_labels)

    if args.do_train:
        train(args, train_dataloader, valid_dataloader, test_dataloader, model)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        model_checkpoint_dir = os.path.join(args.model_checkpoint_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(model_checkpoint_dir, map_location=args.device))
        model.to(args.device)
        test_result = evaluate(args, test_dataloader, model)

        print("***** Test results *****")
        for key in sorted(test_result.keys()):
            print("  {} = {}".format(key, str(round(test_result[key], 4))))

        if args.do_explain:
            correct_lines = get_dep_add_lines_bigvul()
            ipt_save_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}")
            if not os.path.exists(ipt_save_dir):
                os.makedirs(ipt_save_dir)
            if args.hyper_para:
                if args.ipt_method == "cfexplainer":
                    if args.cfexp_L1:
                        ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}_L1_{args.cfexp_alpha}.pt")
                    else:
                        ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}_{args.cfexp_alpha}.pt")
            else:
                ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}.pt")
            print("Size of test dataset:", len(test_data))

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            if not os.path.exists(ipt_save) or args.ipt_update:
                graph_exp_list = []
                if args.ipt_method == "pgexplainer":
                    eval_model = Detector(args)
                    eval_model.load_state_dict(torch.load(model_checkpoint_dir, map_location=args.device))
                    eval_model.to(args.device)
                    graph_exp_list = pgexplainer_run(args, model, eval_model, train_dataset, test_dataset,
                                                     correct_lines)
                elif args.ipt_method == "subgraphx":
                    graph_exp_list = subgraphx_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gnn_lrp":
                    graph_exp_list = gnn_lrp_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "deeplift":
                    graph_exp_list = deeplift_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gradcam":
                    graph_exp_list = gradcam_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gnnexplainer":
                    graph_exp_list = gnnexplainer_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "cfexplainer":
                    graph_exp_list = cfexplainer_run(args, model, test_data, correct_lines)

                torch.save(graph_exp_list, ipt_save)

            eval_exp(ipt_save, model, correct_lines, args)


if __name__ == "__main__":
    main()
