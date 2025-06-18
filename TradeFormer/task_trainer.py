from torch_geometric.data import Batch
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from utils.utils import (
    save_model,
    get_regresssion_metrics,
    LossAnomalyDetector,
)
import torch.nn.functional as F
from sklearn.metrics import classification_report, matthews_corrcoef
import random
from torch.cuda.amp import GradScaler
from grit.LossLib import UncertaintyWeight


class TaskTrainer:
    def __init__(
        self,
        model,
        output_dir,
        grad_norm_clip=1.0,
        device="cuda",
        max_epochs=10,
        use_amp=True,
        task_type="regression",
        learning_rate=1e-4,
        lr_patience=20,
        lr_decay=0.5,
        min_lr=1e-5,
        weight_decay=0.0,
    ):
        self.model = model
        self.output_dir = output_dir
        self.grad_norm_clip = grad_norm_clip
        self.writer = SummaryWriter(self.output_dir)
        self.learning_rate = learning_rate
        self.device = device
        self.n_epochs = max_epochs
        self.use_amp = use_amp
        self.task_type = task_type
        self.loss_anomaly_detector = LossAnomalyDetector()
        self.grad_scaler = GradScaler()
        self.loss_fn2 = nn.BCEWithLogitsLoss(
            weight=torch.tensor([0.2, 1.0], device="cuda")
        )

        if task_type == "regression":
            self.loss_fn = nn.MSELoss()
        elif task_type == "classification":
            self.loss_fn = UncertaintyWeight(2)
        else:
            raise Exception(f"Unknown task type: {task_type}")

        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        self.optimizer = torch.optim.Adam(
            filter(
                lambda x: x.requires_grad,
                list(raw_model.parameters()) + list(self.loss_fn.parameters()),
            ),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_decay,
            patience=lr_patience,
            verbose=True,
        )
        self.min_lr = min_lr

    def fit(self, train_loader, val_loader=None, test_loader=None, save_ckpt=True):
        model = self.model.to(self.device)

        best_loss = np.float32("inf")
        best_appearance = np.float32(0)

        for epoch in range(self.n_epochs):
            torch.cuda.empty_cache()
            train_loss = self.train_epoch(epoch, model, train_loader)
            if val_loader is not None:
                val_loss, mcc_metric = self.eval_epoch(
                    epoch, model, val_loader, e_type="val"
                )

            if test_loader is not None:
                test_loss, _ = self.eval_epoch(epoch, model, test_loader, e_type="test")

            curr_loss = val_loss if "val_loss" in locals() else train_loss
            curr_appearance = mcc_metric

            if (
                self.output_dir is not None and save_ckpt and curr_loss < best_loss
            ):  # only save better loss
                best_loss = curr_loss
                self._save_model(self.output_dir, str(epoch + 1), curr_loss)

            if (
                self.output_dir is not None
                and save_ckpt
                and curr_appearance > best_appearance
            ):
                best_appearance = curr_appearance
                self._save_model(
                    self.output_dir, str(epoch + 1) + "_mcc_", curr_appearance
                )

            if self.optimizer.param_groups[0]["lr"] < float(self.min_lr):
                logger.info("Learning rate == min_lr, stop!")
                break
            self.scheduler.step(val_loss)

        if self.output_dir is not None and save_ckpt:  # save final model
            self._save_model(self.output_dir, "final", curr_loss)

    def run_forward(self, model, batch):
        batch = batch.to(self.device)
        pred, true, hidden_state = model(batch)

        pred = pred.squeeze(-1) if pred.ndim > 1 else pred
        true = true.squeeze(-1) if true.ndim > 1 else true

        zero_indices = torch.all(true == torch.tensor([0.0, 0.0], device="cuda"), dim=1)

        true = true[~zero_indices]
        pred = pred[~zero_indices]

        pred = pred.float()
        loss = self.loss_fn2(pred, true)
        raw_pred = pred.clone()
        pred = torch.sigmoid(pred)

        return loss, pred, true, raw_pred

    def compute_kl_loss(self, p, q):
        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none"
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none"
        )

        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

        loss = (p_loss + q_loss) / 2
        return loss

    @torch.no_grad()
    def erase_up(self, real_batch, ratio=0.15, min_connect=0.9):
        """
        Prune the connections in the graph while ensuring that the overall connectivity is not significantly compromised.

        Args:
            real_batch：The current batch.
            ratio: The coefficient that controls the maximum number of edges that can be removed in the graph.
            min_connect: The edge deletion intensity coefficient. which ensures that the number of edges for each node remains
            at a certain proportion after the erase operation.
        """
        batch = real_batch.clone()
        del real_batch

        batch_size = len(batch)
        datalist = []
        for i in range(batch_size):
            data = batch[i]
            pagerank = data.addition[:, 1]
            log_pr = torch.log(pagerank + 1).unsqueeze(-1)
            pagerank_tensor = log_pr.view(data.num_nodes, 1)
            sampling_probabilities = pagerank_tensor.view(-1)  # shape:-> [num_nodes]

            num_edges = data.num_edges
            num_nodes = data.num_nodes

            out_power = [0] * num_nodes
            in_power = [0] * num_nodes

            e = data.edge_index
            for v in range(num_edges):
                src = e[0, v]
                dst = e[1, v]
                out_power[src] += 1
                in_power[dst] += 1

            erase_limit = num_edges * ratio
            tmp_edges = data.edge_index.clone()
            sample_list = torch.multinomial(
                sampling_probabilities, num_samples=int(erase_limit), replacement=False
            )
            drop_ct = 0
            for samples_dot in sample_list:
                out_edge_index = torch.where(tmp_edges[0] == samples_dot)[0]
                out_count = out_edge_index.shape[0]
                candidate = []
                if out_count > min_connect * out_power[samples_dot]:
                    out_dots = tmp_edges[1][out_edge_index]
                    for dot in out_dots:
                        father_count = torch.where(tmp_edges[1] == dot)[0].shape[0]
                        if father_count > min_connect * in_power[dot]:
                            candidate.append((samples_dot, dot))

                in_edge_index = torch.where(tmp_edges[1] == samples_dot)[0]
                in_count = in_edge_index.shape[0]
                if in_count > min_connect * in_power[samples_dot]:
                    in_dots = tmp_edges[0][in_edge_index]
                    for dot in in_dots:
                        son_count = torch.where(tmp_edges[0] == dot)[0].shape[0]
                        if son_count > min_connect * out_power[dot]:
                            candidate.append((dot, samples_dot))

                if len(candidate) > 0:
                    drop_ct += 1
                    drop_choice = random.randint(0, len(candidate) - 1)
                    drop_e = candidate[drop_choice]
                    head = drop_e[0]
                    tail = drop_e[1]
                    index_to_delete = torch.where(
                        (tmp_edges[0] == head) & (tmp_edges[1] == tail)
                    )[0]
                    tmp_edges = torch.cat(
                        [
                            tmp_edges[:, :index_to_delete],
                            tmp_edges[:, index_to_delete + 1 :],
                        ],
                        dim=1,
                    )

            data.edge_index = tmp_edges
            datalist.append(data)
        return Batch.from_data_list(datalist)

    @torch.no_grad()
    def mix_up(self, real_batch, ratio=0.1, threshold=0.85, alpha=1.0):
        """
        Perform mix-up operations on the features of some nodes in the graph.

        Args:
            real_batch：The current batch.
            ratio: The coefficient that controls the maximum number of nodes that can be mixed up in the graph.
            threshold：The upper limit of search times for the mix-up operation in the graph.
            alpha: The hyperparameter for sampling from the Beta distribution.
        """
        batch = real_batch.clone()
        lam = np.random.beta(alpha, alpha)
        batch_size = len(batch)
        for i in range(batch_size):
            data = batch[i]
            origin_feature = data.x.clone()
            origin_one_hot = data.binary_label.clone()
            origin_index = data.edge_index.clone()

            num_vectors = origin_index.size(1)
            num_select = int(num_vectors * ratio)
            fusion_list = []
            mode = ["top", "bot"]
            searchTime = 0
            while (
                len(fusion_list) < num_select and searchTime <= threshold * num_vectors
            ):
                index = random.randint(0, num_vectors - 1)
                source_dot = origin_index[0][index].item()
                target_dot = origin_index[1][index].item()
                if target_dot in fusion_list and source_dot in fusion_list:
                    searchTime = searchTime + 1
                    continue

                # Search for edges with at least one end labeled
                if data.y[target_dot] == 2 and data.y[source_dot] == 2:
                    searchTime = searchTime + 1
                    continue

                act = random.choices(mode, weights=[1, 1], k=1)

                x_i = origin_feature[target_dot, :]
                x_j = origin_feature[source_dot, :]
                y_i = origin_one_hot[target_dot]
                y_j = origin_one_hot[source_dot]

                # Source point fusion to target point
                if act == "top":
                    if target_dot not in fusion_list:
                        data.x[target_dot, :] = lam * x_i + (1 - lam) * x_j
                        data.binary_label[target_dot] = lam * y_i + (1 - lam) * y_j
                        fusion_list.append(target_dot)
                    else:
                        data.x[source_dot, :] = lam * x_j + (1 - lam) * x_i
                        data.binary_label[source_dot] = lam * y_j + (1 - lam) * y_i
                        fusion_list.append(source_dot)
                # Target point fusion to source point
                else:
                    if source_dot not in fusion_list:
                        data.x[source_dot, :] = lam * x_j + (1 - lam) * x_i
                        data.binary_label[source_dot] = lam * y_j + (1 - lam) * y_i
                        fusion_list.append(source_dot)
                    else:
                        data.x[target_dot, :] = lam * x_i + (1 - lam) * x_j
                        data.binary_label[target_dot] = lam * y_i + (1 - lam) * y_j
                        fusion_list.append(target_dot)
                searchTime = 0
        return batch

    def train_epoch(self, epoch, model, train_loader):
        model.train()
        losses = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for it, batch in pbar:
            if self.device == "cuda":
                with torch.autocast(
                    device_type=self.device, dtype=torch.float16, enabled=self.use_amp
                ):
                    # mode = ['vanilla', 'mix_up', 'erase_up']
                    options = [0, 1, 2]
                    probabilities = [0.5, 0.05, 0.45]
                    choice = np.random.choice(options, p=probabilities)
                    batch = batch.to("cpu")

                    if choice == 0:
                        pass
                    elif choice == 1:
                        batch = self.mix_up(batch, ratio=0.2, threshold=0.9)
                    else:
                        batch = self.erase_up(batch, min_connect=0.85, ratio=0.15)
                    r_batch = batch.clone()
                    batch = batch.cuda()

                    torch.cuda.empty_cache()
                    loss1, _, _, logits1 = self.run_forward(model, batch)

                    r_batch = r_batch.cuda()
                    torch.cuda.empty_cache()
                    loss2, _, _, logits2 = self.run_forward(model, r_batch)

                    kl_loss = self.compute_kl_loss(logits1, logits2)
                    CE_loss = 0.5 * (loss1 + loss2)
                    loss, params = self.loss_fn(
                        kl_loss, CE_loss
                    )  # Using Uncertainty Weight to Adjust Loss Weights

                    loss = loss.mean()

            else:
                loss, _, _, _ = self.run_forward(model, batch)

            if self.loss_anomaly_detector(loss.item()):
                logger.info(
                    f"Anomaly loss detected at epoch {epoch + 1} iter {it}: train loss {loss:.5f}."
                )
                del loss, batch
                continue
            else:
                losses.append(loss.item())
                pbar.set_description(
                    f"epoch {epoch + 1} iter {it}: train loss {loss:.5f}. params {params}"
                )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        loss = float(np.mean(losses))
        logger.info(f"train epoch: {epoch + 1}/{self.n_epochs}, loss: {loss:.4f}")
        self.writer.add_scalar(f"train_loss", loss, epoch + 1)
        return loss

    @torch.no_grad()
    def eval_epoch(self, epoch, model, test_loader, e_type="test"):
        model.eval()
        losses = []
        y_test = []
        y_test_hat = []

        pbar = enumerate(test_loader)
        for it, batch in pbar:
            if self.device == "cuda":
                with torch.autocast(
                    device_type=self.device, dtype=torch.float16, enabled=self.use_amp
                ):
                    loss, y_hat, y, params = self.run_forward(model, batch)
                    loss = loss.mean()
                    losses.append(loss.item())
                    if torch.isnan(loss):
                        logger.info("model output NaN")
                        logger.info(torch.any(torch.isnan(y_hat)))
            else:
                loss, y_hat, y, ____ = self.run_forward(model, batch)
            losses.append(loss.item())
            y_test_hat.append(y_hat.cpu().numpy())
            y_test.append(y.cpu().numpy())

        loss = float(np.mean(losses))

        logger.info(f"{e_type} epoch: {epoch + 1}, loss: {loss:.4f}")
        self.writer.add_scalar(f"{e_type}_loss", loss, epoch + 1)

        y_test = np.concatenate(y_test, axis=0).squeeze()
        y_test_hat = np.concatenate(y_test_hat, axis=0).squeeze()
        y_test_hat = np.argmax(y_test_hat, axis=1)

        if self.task_type == "regression":
            mae, mse, _, spearman, pearson = get_regresssion_metrics(
                y_test_hat, y_test, print_metrics=False
            )
            logger.info(
                f"{e_type} epoch: {epoch + 1}, spearman: {spearman:.3f}, pearson: {pearson:.3f}, mse: {mse:.3f}, mae: {mae:.3f}"
            )
            self.writer.add_scalar("spearman", spearman, epoch + 1)
            metric = spearman
        elif self.task_type == "classification":
            converted_list = []
            for item in y_test:
                if np.array_equal(item, [1, 0]):
                    converted_list.append(0)
                elif np.array_equal(item, [0, 1]):
                    converted_list.append(1)
                elif np.array_equal(item, [0, 0]):
                    converted_list.append(2)
                else:
                    print("err!", item)
            y_test = converted_list
            indice = [i for i, element in enumerate(y_test) if element != 2]

            y_test = [y_test[i] for i in indice]
            y_test_hat = [y_test_hat[i] for i in indice]

            mcc = matthews_corrcoef(y_test, y_test_hat)
            logger.info(
                f"\n{classification_report(y_test, y_test_hat, digits=4)}\n{e_type} epoch: {epoch + 1},  mcc: {mcc:.4f}"
            )

            self.writer.add_scalar(f"{e_type}_mcc", mcc, epoch + 1)
            metric = mcc
        return loss, metric

    def _save_model(self, base_dir, info, valid_loss):
        """Save model with format: model_{info}_{valid_loss}"""
        base_name = f"model_{info}_{valid_loss:.3f}"
        save_model(self.model, base_dir, base_name)
