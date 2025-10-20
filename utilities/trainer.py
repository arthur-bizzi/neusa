import torch.optim as optim

class AdamTrainer:
    def __init__(self,params,adam_lr, adam_steps, learning_rate_scheduler, loss_function, adam_eps=1e-8, adam_amsgrad=False,gamma=0.999):

        self.params = params
        self.loss_function = loss_function

        self.adam_steps = adam_steps
        self.adam_lr = adam_lr
        self.adam_eps = adam_eps
        self.adam_amsgrad = adam_amsgrad

        self.optimizer = optim.Adam(self.params, lr=self.adam_lr, eps=self.adam_eps, amsgrad=adam_amsgrad)

        self.learning_rate_scheduler = learning_rate_scheduler
        self.lr_scheduler_mlp = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.lr_scheduler_neusa = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma, last_epoch=-1)


        self.lr_track = []
        self.loss_track_total = []
        self.loss_track_ic = []
        self.loss_track_bc = []
        self.loss_track_res = []

    
    def train(self):
        for step in range(self.adam_steps):
            self.optimizer.zero_grad()

            losses = self.loss_function()
            loss_ic, loss_bc, loss_res = losses
            total_loss = sum(losses)
            
            total_loss.backward()
            self.optimizer.step()

            if self.learning_rate_scheduler and step % 100 == 0 and step != 0:
                self.lr_scheduler_mlp.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_track.append(current_lr)

            self.loss_track_total.append(total_loss.item())
            self.loss_track_ic.append(loss_ic.item())
            self.loss_track_bc.append(loss_bc.item())
            self.loss_track_res.append(loss_res.item())
            if step % 10 == 0 or step == self.adam_steps - 1:
                print(f"Adam step {step} | Loss: {total_loss.item():.6f}")
        print(f"Adam Optimization Completed. ")
        print(f'Adam training loss: {total_loss.item():.6f}')

    def train_neusa(self):
        for step in range(self.adam_steps):
            self.optimizer.zero_grad()

            loss = self.loss_function()

            loss.backward()
            self.optimizer.step()

            self.lr_scheduler_neusa.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_track.append(current_lr)

            self.loss_track_total.append(loss.item())
            if step % 10 == 0 or step == self.adam_steps - 1:
                print(f"Adam step {step} | Loss: {loss.item():.6f}")
        print(f"Adam Optimization Completed. ")
        print(f'Adam training loss: {loss.item():.6f}')

