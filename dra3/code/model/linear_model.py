import torch, copy
import numpy as np
import dill
from functools import reduce

from env.base_env import BaseGame
from typing import *


class NumpyLinearModel():
    def __init__(
        self, 
        observation_size:tuple[int, int], 
        action_space_size:int, 
        config=None, 
        device:torch.device='cpu',
        base_function:Callable=None,
        ):
        if str(device)!="cpu":
            print(f"[WARNING] Linear model is designed to be runing on CPU only. The device will be set to CPU instead of {device}")
        self.action_size = action_space_size
        self.observation_size = reduce(lambda x, y: x*y , observation_size, 1) if isinstance(observation_size, Iterable) else observation_size
        
        self.base_function = base_function
        if base_function is not None:
            dummy_input = np.zeros((1, self.observation_size))
            self.observation_size = self.base_function(dummy_input).shape[-1]
        
        self.device = device
        self.config = config 
        
        # parameters of policy model
        # pi = softmax(X@w_pi + b_pi)
        self.w_pi = np.zeros((self.observation_size, self.action_size))
        self.b_pi = np.zeros(self.action_size)
        
        # parameters of value model
        self.w_v  = np.zeros(self.observation_size)
        self.b_v  = np.zeros(1)
    
    
        
        
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # shape of X:         (B, O)
        # shape of self.w_pi: (O, A)
        # shape of self.b_pi: (A,)
        # shape of self.w_v:  (O,)
        # shape of self.b_v:  (1,)
        # B = Batch Size
        # O = Observation Size
        # A = Action Space Size
        
        # @return: pi(shape=(B, A)), v(shape=(B, 1))
        if self.base_function is not None:
            X = self.base_function(X)
        if type(X) is torch.Tensor:
            X = X.cpu().numpy()
        X = X.reshape(X.shape[0], -1)
        return self.forward_pi(X), self.forward_v(X) 
    
    def forward_pi(self, X:np.ndarray) -> np.ndarray:
        pi = X@self.w_pi + self.b_pi.reshape(1, -1)
        pi = np.exp(pi)
        pi = pi/np.sum(pi, axis=1).reshape(-1, 1)
        return pi
    
    def forward_v(self, X:np.ndarray) -> np.ndarray:
        w_v = self.w_v.reshape(-1, 1)
        v  = X@w_v  + self.b_v
        v  = np.tanh(v)
        return v
    
    def loss_pi(self, pi_pred:np.ndarray, pi_trgt:np.ndarray) -> float:
        B = pi_trgt.shape[0]
        return -np.sum(np.log(pi_pred)*pi_trgt)/B
    
    def loss_v(self, v_pred:np.ndarray, v_trgt:np.ndarray) -> float:
        ########################
        # TODO: your code here #
        # Compute the loss of value
        # v_pred: model predicted values
        # v_trgt: ground truth values
        # @return: an float value, reprenting the loss
        ########################
        return None
    
    def grad_pi(self, X:np.ndarray, pi_trgt:np.ndarray, pi_pred:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        B = X.shape[0]
        grad_w = X.T@(pi_pred-pi_trgt)/B
        grad_b = np.mean(pi_pred-pi_trgt, axis=0)
        return grad_w, grad_b
    
    def grad_v(self, X:np.ndarray, v_trgt:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ########################
        # TODO: your code here #
        # Compute the gradient of w_v and b_v
        # X: origin input
        # v_trgt: ground truth values
        # @return: Tuple(grad_w_v, grad_b_v)
        ########################
        grad_w_v =  None # shape (O, )
        grad_b_v =  None # shape (1,)
        return grad_w_v, grad_b_v
    
    def weight_decay(self, a:float=.0) -> None:
        a = 1-a
        self.w_pi *= a
        self.w_v  *= a
        self.b_pi *= a
        self.b_v  *= a
    
    def learn(self, lr:float, X:np.ndarray, pi_trgt:np.ndarray, pi_pred:np.ndarray, v_trgt:np.ndarray):
        if self.base_function is not None:
            X = self.base_function(X)
        grad_w_pi, grad_b_pi = self.grad_pi(X, pi_trgt, pi_pred)
        grad_w_v, grad_b_v = self.grad_v(X, v_trgt)
        self.w_pi += - grad_w_pi * lr
        self.b_pi += - grad_b_pi * lr
        self.w_v  += - grad_w_v  * lr
        self.b_v  += - grad_b_v  * lr
        
    def save(self, path:str):
        params = {
            "w_pi":self.w_pi,
            "b_pi":self.b_pi,
            "w_v":self.w_v,
            "b_v":self.b_v,
            "observation_size":self.observation_size,
            "action_size":self.action_size,
            "config":self.config,
            "device":self.device,
            "base_function":self.base_function
        }
        with open(path, "wb") as f:
            dill.dump(params, f, recurse=True)
    
    def load(self, path:str):
        with open(path, "rb") as f:
            params = dill.load(f)
        for k, v in params.items():
            self.__setattr__(k, v)
        
        
    
def check_grad(model:NumpyLinearModel):
    # Test whether the gradient calculation is correct
    # If correct, all [Max Error] value should be <= 0.000001
    # NOTE: This function only verifies that the gradient conforms 
    #      to the loss function you defined, and does not guarantee 
    #      that the loss function itself is calculated correctly
    
    t = 1e-8
    B = 128
    np.random.seed(10086)
    
    # genrate random data
    X = np.random.random((B, model.observation_size))
    y = np.random.random((B, 1))
    
    model.w_v = (np.random.random(model.w_v.shape)-0.5)/np.prod(model.w_v.shape)
    model.b_v =  np.random.random(model.b_v.shape)-0.5
    model.w_pi = (np.random.random(model.w_pi.shape)-0.5)/np.prod(model.w_pi.shape)
    model.b_pi = (np.random.random(model.b_pi.shape)-0.5)/np.prod(model.b_pi.shape)
    
    #######################
    # Check Gradient of w_v&b_v
    # back up model& param
    model = copy.deepcopy(model)
    old_w_v = model.w_v
    old_b_v = model.b_v
    
    approx_grad_w_v = np.zeros_like(old_w_v)
    approx_grad_b_v = np.zeros_like(old_b_v)
    
    # compute grad of w_v
    for i in range(approx_grad_w_v.shape[0]):
        model.w_v = old_w_v
        model.w_v[i] -= t
        loss_1 = model.loss_v(model.forward_v(X), y)
        model.w_v[i] += t*2
        loss_2 = model.loss_v(model.forward_v(X), y)
        approx_grad_w_v [i]  =  (loss_2-loss_1)/(t*2)
    #compute grad of b_v
    model.w_v = old_w_v
    model.b_v -= t
    loss_1 = model.loss_v(model.forward_v(X), y)
    model.b_v += t*2
    loss_2 = model.loss_v(model.forward_v(X), y)
    approx_grad_b_v  =  (loss_2-loss_1)/(t*2)
    
    model_grad_w_v, model_grad_b_v = model.grad_v(X, y)
    error_grad_w_v = np.abs(approx_grad_w_v - model_grad_w_v)
    error_grad_b_v = np.abs(approx_grad_b_v - model_grad_b_v)
    print(f"[Gradient of w_v] shape:{error_grad_w_v.shape}   \tMean Error: {np.mean(error_grad_w_v):.6f}, Max Error: {np.max(error_grad_w_v):.6f}")
    print(f"[Gradient of b_v] shape:{error_grad_b_v.shape}        \tMean Error: {np.mean(error_grad_b_v):.6f}, Max Error: {np.max(error_grad_b_v):.6f}")
    
    ######################
    # Check Gradient of w_pi&b_pi
    # back up model& param
    model = copy.deepcopy(model)
    
    old_w_pi = model.w_pi
    old_b_pi = model.b_pi
    
    y = np.zeros((B, model.action_size))
    # for i in range(B):
    #     y[i, np.random.randint(model.action_size)]=1
    y = np.random.random((B, model.action_size))
    y = y/np.sum(y, axis=1).reshape(-1, 1)
    
    approx_grad_w_pi = np.zeros_like(old_w_pi)
    approx_grad_b_pi = np.zeros_like(old_b_pi)
    
    # compute grad of w_pi
    for i in range(approx_grad_w_pi.shape[0]):
        for j in range(approx_grad_w_pi.shape[1]):
            model.w_pi = old_w_pi
            model.w_pi[i,j] -= t
            loss_1 = model.loss_pi(model.forward_pi(X), y)
            model.w_pi[i,j] += t*2
            loss_2 = model.loss_pi(model.forward_pi(X), y)
            approx_grad_w_pi [i, j]  =  (loss_2-loss_1)/(t*2)
    #compute grad of b_pi
    model.w_pi = old_w_pi
    for i in range(approx_grad_b_pi.shape[0]):
        model.b_pi[i] -= t
        loss_1 = model.loss_pi(model.forward_pi(X), y)
        model.b_pi[i] += t*2
        loss_2 = model.loss_pi(model.forward_pi(X), y)
        approx_grad_b_pi[i]  =  (loss_2-loss_1)/(t*2)
    
    model_grad_w_pi, model_grad_b_pi = model.grad_pi(X, y, model.forward_pi(X))
    error_grad_w_pi = np.abs(approx_grad_w_pi - model_grad_w_pi)
    error_grad_b_pi = np.abs(approx_grad_b_pi - model_grad_b_pi)
    print(f"[Gradient of w_pi] shape:{error_grad_w_pi.shape}   \tMean Error: {np.mean(error_grad_w_pi):.6f}, Max Error: {np.max(error_grad_w_pi):.6f}")
    print(f"[Gradient of b_pi] shape:{error_grad_b_pi.shape}   \tMean Error: {np.mean(error_grad_b_pi):.6f}, Max Error: {np.max(error_grad_b_pi):.6f}")
    

if __name__ == "__main__":
    # Test gradient calculation
    model = NumpyLinearModel(247, 49)
    check_grad(model)
    
    
    
        