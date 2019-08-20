from imports import *

def save_model(m, p): torch.save(m.state_dict(), p)


def load_model(m, p): m.load_state_dict(torch.load(p))


def softmax(x):
    m = x.max(1)
    num = np.exp(x-np.expand_dims(m,1))
    den = np.exp(x-np.expand_dims(m,1)).sum(1)
    return num/np.expand_dims(den,1)


def accuracy(y, pred):
    """
    Accuracy score
    """
    pred = pred.argmax(-1)
    return accuracy_score(y, pred)


def cos_cycle(start_lr, end_lr, n_iterations):
    '''cosine annealing'''
    i = np.arange(n_iterations)
    c_i = 1 + np.cos(i*np.pi/n_iterations)
    return end_lr + (start_lr - end_lr)/2 *c_i


class step_policy:
    '''
    One-cycle learning rate and momentum policy with cosine annealing.
    '''


    def __init__(self, n_epochs, dl, max_lr, div_factor:float=25., pctg:float=.3, moms:tuple=(.95,.85), delta=1/1e4):
        
        total_iterations = n_epochs*len(dl)
        
        max_lr, min_start, min_end = (max_lr, 
                                      max_lr/div_factor, 
                                      max_lr/div_factor*delta)
        
        self.stages = (int(total_iterations*pctg), total_iterations - int(total_iterations*pctg))
        
        lr_diffs = ((min_start, max_lr),(max_lr, min_end))
        mom_diffs = (moms, (moms[1],moms[0]))

        self.lr_schedule = self._create_schedule(lr_diffs)
        self.mom_schedule = self._create_schedule(mom_diffs)
        
        self.iter = -1


    def _create_schedule(self, diffs):
        individual_stages = [cos_cycle(start, end, n) for ((start, end),n) in zip(diffs, self.stages)]
        return np.concatenate(individual_stages)


    def step(self):
        self.iter += 1
        return [sch[self.iter] for sch in [self.lr_schedule, self.mom_schedule]]
    
    
class OptimizerWrapper:
    '''
    Wrapper to use wight decay in optim.Adam without influencing its algorithm.
    Takes care of the change in learning rate / momenutm at every iteration.
    
    '''


    def __init__(self, model, n_epochs, dl, max_lr, div_factor=None, wd=0):
        
        self.policy =  step_policy(n_epochs=n_epochs, dl=dl, 
                                   max_lr=max_lr, div_factor=div_factor)
        
        self.model = model
        self._wd = wd
        
        p = filter(lambda x: x.requires_grad, model.parameters())
        
        self.optimizer = optim.Adam(params=p, lr=0)


    def _update_optimizer(self):
        lr_i, mom_i = self.policy.step()
        for group in self.optimizer.param_groups:
            group['lr'] = lr_i
            group['betas'] = (mom_i, .999)


    def step(self):
        self._update_optimizer()
        if self._wd!=0:
            for group in self.optimizer.param_group:
                for p  in group['params']: p.data.mul_(group['lr']*self._wd)
        self.optimizer.step()


    def zero_grad(self): self.optimizer.zero_grad()


    def reset(self, n_epochs, dl, max_lr):
        self.iter = -1
        self.policy =  step_policy(n_epochs=n_epochs, dl=dl, max_lr=max_lr)


def validate(model, valid_dl, h_0):
    """
    Validation/Testing loop
    """
    model.eval()
    div = 0
    agg_loss = 0
    ys = np.empty((0), int)
    preds = np.empty((0, 10), float)
    for it, (x,y) in enumerate(valid_dl):
        
        x = x.long().cuda()
        y = y.long().cuda()
        
        out = model(x, h_0)
        loss = F.cross_entropy(input=out,target=y)

        agg_loss += loss.item()
        div += 1
        
        preds = np.append(preds, out.cpu().detach().numpy(), axis=0)
        ys = np.append(ys, y.cpu().numpy(), axis=0)
    
    val_loss = agg_loss/div
    measures = accuracy(ys, softmax(preds))
    model.train()
    return val_loss, measures
      

def train(n_epochs, train_dl, model, h_0, valid_dl=None, max_lr=.01, div_factor=25):
    """Training loop
    """
            
    optimizer = OptimizerWrapper(model, n_epochs, train_dl,
                                 max_lr=max_lr, div_factor=div_factor)
    
    min_val_loss = np.inf
    
    for epoch in tqdm(range(n_epochs)):
        model.train()
        div = 0
        agg_loss = 0
        for it, (x,y) in enumerate(train_dl):
            
            x = x.long().cuda()
            y = y.long().cuda()
            
            out = model(x, h_0)
            loss = F.cross_entropy(input=out,target=y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            agg_loss += loss.item()
            div += 1
            
        if valid_dl is None: print(f'Ep. {epoch+1} - train loss {agg_loss/div:.4f}')
        else:
            val_loss, measure = validate(model, valid_dl, h_0)
            print(f'Ep. {epoch+1} - train loss {agg_loss/div:.4f} -  val loss {val_loss:.4f} avg accuracy {measure:.4f}')
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_model(model, './best_model.pth')
                torch.save(h_0, 'best_model_hs.pth')
