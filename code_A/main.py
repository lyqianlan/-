import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
# 生成环形数据集（无噪声）
X, y = make_circles(n_samples=1000, noise=0.0, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# 定义模型构建函数
def build_model(hidden_layers, neurons_per_layer, activation='relu'):
    model = tf.keras.Sequential()
    for i in range(hidden_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
# 实验1：网络结构对比
configs = [
    (1, 2), (1, 4), (2, 4), (2, 8), (3, 4), (3, 8)
]
results = []
for layers, neurons in configs:
    print(f"\nTraining {layers} layer(s) x {neurons} neurons")
    model = build_model(layers, neurons, activation='relu')
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    train_loss = history.history['loss'][-1]
    test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
    results.append((layers, neurons, train_loss, test_loss))
    print(f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
# 输出表格
print("\n===== Table 1 =====")
print("Layers\tNeurons\tTrain Loss\tTest Loss")
for r in results:
    print(f"{r[0]}\t{r[1]}\t{r[2]:.4f}\t\t{r[3]:.4f}")
# 实验2：激活函数对比
activations = ['relu', 'tanh', 'sigmoid', 'linear']
for act in activations:
    print(f"\nTraining with activation = {act}")
    model = build_model(3, 4, activation=act)
    history = model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0)
    train_loss = history.history['loss'][-1]
    test_loss = model.evaluate(X_test, y_test, verbose=0)[0]
    print(f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
# 可视化决策边界（以最佳模型为例）
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
    Z = (Z > 0.5).astype(int).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title(title)
    plt.show()
best_model = build_model(3, 4, activation='relu')
best_model.fit(X_train, y_train, epochs=100, verbose=0)
plot_decision_boundary(best_model, X_test, y_test, "Decision Boundary (3x4, ReLU)")