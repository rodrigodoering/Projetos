

class Perceptron():
    
    """
    Esse objeto implementa um Neurônio Perceptron simples com uma função de ativação
    E tem como único método "predict" que retorna um label dado os inputs e coeficientes
    
    Atributos:
    self.W = np.array, matriz de pesos n_inputs x 1
    self.bias = np.array, bias do modelo
    self.g = função de ativação do modelo, por padrão pe implementada uma step function com threshold = 0
    """
    
    def __init__(self, n_inputs=False, W=False, b=False, activation=lambda x: np.array([1 if z > 0 else 0 for z in x])):
        self.W = W
        self.bias = b
        self.g = activation
    
    
    def predict(self, X):
        """
        X: Numpy.array de dimensões (n_samples x n_features)
        retorna: Ativação do perceptron, dado por g(z)
        A ativação padrão é a step function, retornando 1 se o valor for maior que 0, senão 0
        """ 
        # realiza a função de soma Z = WX + B
        z = np.dot(X, self.W) + self.bias
        
        # retorna a ativação da somatória g(WX + B)
        return self.g(z)
   

	def update(self, x, y):
	    a = self.predict([x])[0]
	    if y == 1 and a == 0:
	        self.W += x
	    elif y == 0 and a == 1:
	        self.W -= x
	    else:
	        pass 