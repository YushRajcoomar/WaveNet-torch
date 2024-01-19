import torch

class WaveNetTrain:
    def __init__(self,model,optimizer,loss_func) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def fit(self,x,y):
        self.optimizer.zero_grad()
        pred = self.model(x.to(self.device))
        label = y.to(self.device)
        loss = self.loss_func(pred,label)

        loss.backward()
        self.optimizer.step()

    def predict(self,x):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x.to(self.device))
        self.model.train()
        return pred.cpu().numpy()