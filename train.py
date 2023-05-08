import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

    def train(self, epoch):
        self.FuFeNet_I.cuda().eval()
        self.FeatNet_I.cuda().eval()#

        self.FuCoNet_I.cuda().train()
        self.CodeNet_I.cuda().train()#
        self.CodeNet_T.cuda().train()
        self.gcn_I.cuda().train()
        self.gcn_T.cuda().train()

        self.FuCoNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)
        self.gcn_I.set_alpha(epoch)
        self.gcn_T.set_alpha(epoch)

        self.logger.info('Epoch [%d/%d], alpha for FusionNet: %.3f, alpha for TxtNet: %.3f' % (
            epoch + 1, settings.NUM_EPOCH, self.FuCoNet_I.alpha, self.CodeNet_T.alpha))

        # Extract the img, txt features
        # featuresimg, featurestxt = self.extract_features(self.train_loader, self.num_features)

        # Generate similarity matrix
        # I_S = self.generate_similarity_matrix(featuresimg, 2, 2).to(torch.device("cuda:0"))
        # T_S = self.generate_similarity_matrix(featurestxt, 2, 2).to(torch.device("cuda:0"))
        criterion = S_Loss()

        loss_func = ContrastiveLoss(settings.BATCH_SIZE)
        for idx, (img, txt, labels, index) in enumerate(self.train_loader):
            batch_size = img.size(0)
            img = Variable(img.cuda())
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())
            self.opt_FI.zero_grad()

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            self.opt_GI.zero_grad()
            self.opt_GT.zero_grad()

            (_, F_I), _, _, _= self.FeatNet_I(img)
            F_T = txt
            _, hid_I, code_I, decoded_t = self.CodeNet_I(img)
            _, hid_T, code_T, decoded_i = self.CodeNet_T(txt)
            F_I = F.normalize(F_I)
            S_I = F_I.mm(F_I.t())
            S_I = S_I * 2 - 1
            F_T = F.normalize(F_T)
            S_T = F_T.mm(F_T.t())
            S_T = S_T * 2 - 1

            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)

            BI_BI = B_I.mm(B_I.t())
            BT_BT = B_T.mm(B_T.t())
            BI_BT = B_I.mm(B_T.t())


            S_tilde = settings.ALPHA * S_I + (1 - settings.ALPHA) * S_T
            S = settings.K * S_tilde

            lossf1 = F.mse_loss(BT_BT, S)
            lossf2 = F.mse_loss(BI_BT, S)
            lossf3 = F.mse_loss(BI_BI, S)
            lossf31 = F.mse_loss(BI_BI, settings.K * S_I)
            lossf32 = F.mse_loss(BT_BT, settings.K * S_T)

            diagonal = BI_BT.diagonal()
            all_1 = torch.rand((batch_size)).fill_(1).cuda()
            lossf4 = F.mse_loss(diagonal, settings.K * all_1)
            lossf5 = F.mse_loss(decoded_i, F_I)
            lossf6 = F.mse_loss(decoded_t, F_T)
            lossf7 = F.mse_loss(BI_BT, BI_BT.t())
            lossf = 1 * lossf1 + 1 * lossf2 + 1 * lossf3 + 1 * lossf4 + 1 * lossf5 + 1 * lossf6 + 2 * lossf7 + settings.ETA * (
                    lossf31 + lossf32)

            (_, F_FI), _, _, _, _, _, _, _, _ = self.FuFeNet_I(img, txt)
            F_FT = txt
            # _, hid_I, code_I, decoded_t = self.CodeNet_I(img)
            # T_, hid_T, code_T, decoded_i = self.CodeNet_T(txt)

            _, hid_FI, code_FI, decoded_Ft, T_, hid_FT, code_FT, decoded_Fi, fusion_f = self.FuCoNet_I(img, txt)
            # kl = F.kl_div(code_I.softmax(dim=-1).log(), code_T.softmax(dim=-1), reduction='none')
            # The similarity of H_I, H_T
            F_G = T_
            H_I = code_FI @ code_FI.t() / settings.CODE_LEN
            H_T = code_FT @ code_FT.t() / settings.CODE_LEN
            with torch.no_grad():
                _, _, _, _, _, _, _, _, fusion_f = self.FuCoNet_I(img, txt)
                Fusion_f = fusion_f.cpu()
                S_F = self.generate_similarity_matrix(Fusion_f, 2, 2).to(torch.device("cuda:0"))

            # targets_I = I_S[index, :][:, index]
            # targets_T = T_S[index, :][:, index]
            #
            # loss_I = criterion(H_I, targets_I)
            # loss_T = criterion(H_T, targets_I)
            F_G = F.normalize(F_G)
            F_FI = F.normalize(F_FI)
            S_FI = F_FI.mm(F_FI.t())
            S_FI = S_FI * 2 - 1
            F_FT = F.normalize(F_FT)
            S_FT = F_FT.mm(F_FT.t())
            S_FT = S_FT * 2 - 1

            T_ = F.normalize(T_)
            S_t = T_.mm(T_.t())
            S_t = S_t * 2 - 1
            fusion_f = F.normalize(fusion_f)
            s_f = fusion_f.mm(fusion_f.t())
            s_f = s_f * 2 - 1

            loss = F.mse_loss(S_FI, S_t)

            B_FI = F.normalize(code_FI)
            B_FT = F.normalize(code_FT)


            BI_FBI = B_FI.mm(B_FI.t())
            BT_FBT = B_FT.mm(B_FT.t())
            BI_FBT = B_FI.mm(B_FT.t())

            BIT = B_I.mm(B_FT.t())
            BTI = B_T.mm(B_FI.t())
            # todo BFII
            BFII = B_I.mm(B_FI.t())
            BFTT = B_T.mm(B_FT.t())

            C = torch.cat((2 * F_FI, 2 * F_FT), 1)
            dis_C = euclidean_dist(C, C)
            A_C = torch.exp(-dis_C / 4)

            C = C.mm(C.t()) * A_C

            in_aff, out_aff = self.normalize(C.type(torch.FloatTensor))
            C = C * 2 - 1

            B_GI = self.gcn_I(F_FI, in_aff, out_aff)
            B_GT = self.gcn_T(F_G, in_aff, out_aff)

            SF_tilde = settings.ALPHA * S_FI + (1 - settings.ALPHA) * S_FT
            # S = settings.K * (S_tilde + s_f)
            # SF = settings.K * SF_tilde + S_F
            # SF = SF_tilde
            # SF = SF + 0.7 * C
            SF = SF_tilde + s_f

            loss1 = F.mse_loss(BT_FBT, SF)
            loss2 = F.mse_loss(BI_FBT, SF)
            loss3 = F.mse_loss(BI_FBI, SF)
            lossg1 = F.mse_loss(B_FI, B_GI)
            lossg2 = F.mse_loss(B_FT, B_GT)
            loss31 = F.mse_loss(BI_FBI, settings.K * S_FI)
            loss32 = F.mse_loss(BT_FBT, settings.K * S_FT)
            loss8 = F.mse_loss(BT_FBT, BI_FBI)
            # loss9 = F.mse_loss(F_I, T_)
            loss10 = F.mse_loss(BT_FBT, BI_FBI)
            loss11 = F.mse_loss(SF, SF.t())
            loss12 = F.mse_loss(BI_FBT, BI_FBI)
            loss13 = F.mse_loss(BI_FBT, BT_FBT)
            loss14 = F.mse_loss(settings.K * S_FI, settings.K * S_FT)

            diagonal = BI_FBT.diagonal()
            all_1 = torch.rand((batch_size)).fill_(1).cuda()
            loss4 = F.mse_loss(diagonal, settings.K * all_1)
            loss5 = F.mse_loss(decoded_Fi, F_FI)
            loss6 = F.mse_loss(decoded_Ft, F_FT)
            loss7 = F.mse_loss(BI_FBT, BI_FBT.t())
            # todo
            diagonal = BFII.diagonal()
            all_1 = torch.rand((batch_size)).fill_(1).cuda()
            lossf4 = F.mse_loss(diagonal, settings.K * all_1)
            diagonal = BFTT.diagonal()
            all_1 = torch.rand((batch_size)).fill_(1).cuda()
            lossf5 = F.mse_loss(diagonal, settings.K * all_1)
            loss_f1 = F.mse_loss(code_I, code_FI)
            loss_f2 = F.mse_loss(code_T, code_FT)
            loss_f3 = F.mse_loss(BI_BT, BI_FBT)

            loss_cont = loss_func(code_I, code_FI)
            loss_cont2 = loss_func(code_T, code_FT)
          
            # loss = lossg1 + lossg2 + loss_cont + loss_cont2 + loss_f1 + loss_f2 + loss_f3 + lossf4 + lossf5 + lossf + loss + loss14 + loss13 + loss12 + loss11 + loss10 + loss8 + 1 * loss1 + 1 * loss2 + 1 * loss3 + 1 * loss4 + 1 * loss5 + 1 * loss6 + 2 * loss7 + settings.ETA * (
            #         loss31 + loss32)
            loss = lossf+1 * loss1 + 1 * loss2 + 1 * loss3 + 1 * loss4 + 1 * loss5 + 1 * loss6 + 2 * loss7 + settings.ETA * (
                    loss31 + loss32)

           
            loss.backward()
            self.opt_FI.step()
            self.opt_I.step()
            self.opt_T.step()
            self.opt_GI.step()
            self.opt_GT.step()
            if (idx + 1) % (len(self.train_dataset) // settings.BATCH_SIZE / settings.EPOCH_INTERVAL) == 0:
                self.logger.info(
                    'Epoch [%d/%d], Iter [%d/%d] '
                    'Loss1: %.4f Loss2: %.4f Loss3: %.4f '
                    'Loss4: %.4f '
                    'Loss5: %.4f Loss6: %.4f '
                    'Loss7: %.4f '
                    'Loss8: %.4f '
                    'Loss10: %.4f '
                    'Loss11: %.4f '
                    'Loss12: %.4f '
                    'Loss13: %.4f '
                    'Loss14: %.4f '
                    'Total Loss: %.4f'
                    % (
                        epoch + 1, settings.NUM_EPOCH, idx + 1,
                        len(self.train_dataset) // settings.BATCH_SIZE,
                        loss1.item(), loss2.item(), loss3.item(),
                        loss4.item(),
                        loss5.item(), loss6.item(),
                        loss7.item(),
                        loss8.item(),
                        loss10.item(),
                        loss11.item(),
                        loss12.item(),
                        loss13.item(),
                        loss14.item(),
                        loss.item()))
                self.All_loss.append(loss.item())
                self.All_epoch.append(epoch+1)
        np.savez('HILN_WIKI_loss.npz', all_loss=self.All_loss, all_epoch=self.All_epoch)
    def eval(self, step=0, last=False):

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.FuCoNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()
        if settings.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(self.database_loader, self.test_loader,
                                                                   self.FuCoNet_I, self.CodeNet_T,
                                                                   self.database_dataset, self.test_dataset)
            K = [1, 200, 400, 500, 1000, 1500, 2000]
        if settings.DATASET == "MIRFlickr" or settings.DATASET == "NUSWIDE":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.FuCoNet_I,
                                                              self.CodeNet_T, self.database_dataset, self.test_dataset)
            K = [1, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
        if settings.EVAL:
            MAP_I2T = calculate_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L)
            MAP_T2I = calculate_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L)
            self.logger.info('--------------------Evaluation: Calculate MAP-------------------')
            self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
            self.logger.info('--------------------------------------------------------------------')
            retI2T = p_topK(qu_BI, re_BT, qu_L, re_L, K)
            retT2I = p_topK(qu_BT, re_BI, qu_L, re_L, K)
            self.logger.info(retI2T)
            self.logger.info(retT2I)
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        # pr = pr_curveIT(qF=qu_BI,rF=re_BT,qL=qu_L,rL=re_L, what=1, topK=5000)
        # pr = pr_curveTI(qF=qu_BT,rF=re_BI,qL=qu_L,rL=re_L, what=1, topK=3000)
        self.logger.info('--------------------Evaluation: Calculate top MAP-------------------')
        self.logger.info('MAP of Image to Text: %.3f, MAP of Text to Image: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('--------------------------------------------------------------------')
        if MAP_I2T + MAP_T2I > self.best and not settings.EVAL:
            self.save_checkpoints(step=step, best=True)
            self.best = MAP_T2I + MAP_I2T
            self.logger.info("#########is best:%.3f #########" % self.best)

    def save_checkpoints(self, step, file_name='%s_%d_bit_latest.pth' % (settings.DATASET, settings.CODE_LEN),
                         best=False):
        if best:
            file_name = '%s_%d_bit_best_epoch.pth' % (settings.DATASET, settings.CODE_LEN)
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.FuCoNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
            'step': step,
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name='%s_%d_bit_best_epoch.pth' % (settings.DATASET, settings.CODE_LEN)):
        ckp_path = osp.join(settings.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.FuCoNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])
        self.logger.info('********** The loaded model has been trained for %d epochs.*********' % obj['step'])

    def normalize(self, affnty):
        col_sum = affnty.sum(axis=1)[:, np.newaxis]
        row_sum = affnty.sum(axis=0)

        out_affnty = affnty / col_sum
        in_affnty = (affnty / row_sum).t()

        out_affnty = Variable(torch.Tensor(out_affnty)).cuda()
        in_affnty= Variable(torch.Tensor(in_affnty)).cuda()

        return in_affnty, out_affnty

    def extract_features(self, train_lodaer, num_features):
        featuresimg = torch.zeros(self.train_loader.dataset.train_labels.shape[0], self.num_features)
        featurestxt = torch.zeros(self.train_loader.dataset.train_labels.shape[0], self.num_features)
        with torch.no_grad():
            N = len(self.train_loader)
            for i, (img, txt, labels, index) in enumerate(self.train_loader):
                img = Variable(img.cuda())
                txt = Variable(torch.FloatTensor(txt.numpy()).cuda())
                (_, F_I), _, _, _, T_, hid_T, code_T, decoded_i = self.FuFeNet_I(img, txt)
                # T_, hid_T, code_T, decoded_i = self.CodeNet_T(txt)
                featuresimg[index, :] = F_I.cpu()
                featurestxt[index, :] = T_.cpu()

        return featuresimg, featurestxt

    def generate_similarity_matrix(self, features, alpha, beta):
        # Cosine similarity
        cos_dist = squareform(pdist(features.numpy(), 'cosine'))

        # Find maximum count of cosine distance
        max_cnt, max_cos = 0, 0
        interval = 1. / 100
        cur = 0
        for i in range(100):
            cur_cnt = np.sum((cos_dist > cur) & (cos_dist < cur + interval))
            if max_cnt < cur_cnt:
                max_cnt = cur_cnt
                max_cos = cur
            cur += interval

        # Split features into two parts
        flat_cos_dist = cos_dist.reshape((-1, 1))
        left = flat_cos_dist[np.where(flat_cos_dist <= max_cos)[0]]
        right = flat_cos_dist[np.where(flat_cos_dist > max_cos)[0]]

        # Reconstruct gaussian distribution
        left = np.concatenate([left, 2 * max_cos - left])
        right = np.concatenate([2 * max_cos - right, right])

        # Model data using gaussian distribution
        left_mean, left_std = norm.fit(left)
        right_mean, right_std = norm.fit(right)

        # Construct similarity matrix
        S = (cos_dist < (left_mean - alpha * left_std)) * 1.0 + (cos_dist > (right_mean + beta * right_std)) * -1.0

        return torch.FloatTensor(S)

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

class S_Loss(nn.Module):
    def __init__(self):
        super(S_Loss, self).__init__()

    def forward(self, H, S):
        loss = (S.abs() * (H - S).pow(2)).sum() / (H.shape[0] ** 2)

        return loss


def main():
    sess = Session()

    if settings.EVAL == True:
        sess.load_checkpoints()
        sess.eval()

    else:
        for epoch in range(settings.NUM_EPOCH):
            # train the Model
            sess.train(epoch)
            # eval the Model
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                sess.eval(step=epoch + 1)
            # save the model
        settings.EVAL = True
        sess.logger.info('---------------------------Test------------------------')
        sess.load_checkpoints()
        sess.eval()


if __name__ == '__main__':
    main()
