import numpy as np 
a=int(input("Quelle période vous plait le plus pour écouter de la musique, entre 0 et 10 ? 0 = avant les années 70', 2 = 70', 4 = 80', 6 = 90', 8 = 2000, 10 = actuelle"))
b=int(input("Ecoutez-vous de la musique pour méditer, entre 0 et 10 ? 0 = oui pour dormir, 5 = yoga/médiation, 10 = pas du tout"))
c=int(input("Ecoutez-vous de la musique pour faire la fête, entre 0 et 10 ? 0 = pas du tout je préfère la bibliothèque, 5 = soirée chill, 10 = GROSSE CAISSE"))
d=int(input("Quel tempo vous préférez vous, entre 0 et 10 ? 0 = lent, 5 = moyen, 10 = rapide"))
e=int(input("Quelle est votre humeur actuelle, entre 0 et 10 ? 0 = triste, 2 = rupture amoureuse, 4 = colère, 6 = appaisé/chill, 8 = de bonne humeur, 10 = JOIE EXTREME"))
f=int(input("En quelle langue écoutez-vous votre musique, entre 0 et 10 ? 2 = pas de parole, 4 = français, 6 = anglais, 8 = espagnol, 10 = autres"))

x_entrer = np.array(([0, 0, 0, 0, 0, 2], [0, 1, 1, 2, 5, 2], [0, 3, 3, 4, 6, 2], [0, 5, 4, 6, 8, 2], [2, 2, 2, 2, 0, 6], [4, 10, 10, 5, 10, 4], [0, 5, 5, 4, 8, 6], [6, 3, 4, 3, 3, 4], [4, 6, 4, 5, 3, 4], [6, 8, 5, 6, 7, 6], [8, 10, 6, 7, 8, 8], [10, 10, 7, 8, 9, 10], [8, 7, 6, 5, 2, 6],[10, 8, 7, 7, 5, 10],[8, 9, 8, 8, 8, 6],[10, 10, 10, 6, 10, 10],[6, 6, 6, 6, 0, 4],[8, 8, 8, 8, 5, 6],[10, 10, 9, 9, 8, 8],[10, 9, 10, 10, 10, 10],[4, 8, 6, 7, 6, 4],[6, 9, 8, 8, 8, 6],[8, 10, 9, 9, 9, 8],[10, 8, 10, 10, 10, 6]), dtype=float) # données d'entrer
y = np.array(([0],[0],[0],[0],[0,2],[0,2],[0,2],[0,2],[0,4],[0,4],[0,4],[0,4],[0,6],[0,6],[0,6],[0,6],[0,8],[0,8],[0,8],[0,8],[1],[1],[1],[1]), dtype=float) # données de sortie /  1 = rouge /  0 = bleu

# Changement de l'échelle de nos valeurs pour être entre 0 et 1
x_entrer = x_entrer/np.amax(x_entrer, axis=0) # On divise chaque entré par la valeur max des entrées

# On récupère ce qu'il nous intéresse
X = np.split(x_entrer, [8])[0] # Données sur lesquelles on va s'entrainer, les 8 premières de notre matrice
xPrediction = np.split(x_entrer, [8])[1] # Valeur que l'on veut trouver

#Notre classe de réseau neuronal
class Neural_Network(object):
  def __init__(self):
        
  #Nos paramètres
    self.inputSize = 2 # Nombre de neurones d'entrer
    self.outputSize = 1 # Nombre de neurones de sortie
    self.hiddenSize = 3 # Nombre de neurones cachés

  #Nos poids
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3) Matrice de poids entre les neurones d'entrer et cachés
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) Matrice de poids entre les neurones cachés et sortie


  #Fonction de propagation avant
  def forward(self, X):

    self.z = np.dot(X, self.W1) # Multiplication matricielle entre les valeurs d'entrer et les poids W1
    self.z2 = self.sigmoid(self.z) # Application de la fonction d'activation (Sigmoid)
    self.z3 = np.dot(self.z2, self.W2) # Multiplication matricielle entre les valeurs cachés et les poids W2
    o = self.sigmoid(self.z3) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
    return o

  # Fonction d'activation
  def sigmoid(self, s):
    return 1/(1+np.exp(-s))

  # Dérivée de la fonction d'activation
  def sigmoidPrime(self, s):
    return s * (1 - s)

  #Fonction de rétropropagation
  def backward(self, X, y, o):

    self.o_error = y - o # Calcul de l'erreur
    self.o_delta = self.o_error*self.sigmoidPrime(o) # Application de la dérivée de la sigmoid à cette erreur

    self.z2_error = self.o_delta.dot(self.W2.T) # Calcul de l'erreur de nos neurones cachés 
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # Application de la dérivée de la sigmoid à cette erreur

    self.W1 += X.T.dot(self.z2_delta) # On ajuste nos poids W1
    self.W2 += self.z2.T.dot(self.o_delta) # On ajuste nos poids W2

  #Fonction d'entrainement 
  def train(self, X, y):
        
    o = self.forward(X)
    self.backward(X, y, o)

  #Fonction de prédiction
  def predict(self):
        
    print("Donnée prédite apres entrainement: ")
    print("Entrée : \n" + str(xPrediction))
    print("Sortie : \n" + str(self.forward(xPrediction)))

    if(self.forward(xPrediction) < 0.5):
        print("La fleur est BLEU ! \n")
    else:
        print("La fleur est ROUGE ! \n")


NN = Neural_Network()

for i in range(1000): #Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
    print("# " + str(i) + "\n")
    print("Valeurs d'entrées: \n" + str(X))
    print("Sortie actuelle: \n" + str(y))
    print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(X),2)))
    print("\n")
    NN.train(X,y)

NN.predict()
