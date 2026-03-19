import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pennylane as qml
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

pd.set_option('future.no_silent_downcasting', True)
jax.config.update("jax_enable_x64", True)

# --- 1. Veri Hazırlığı (Aynı kalıyor) ---
def load_data():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    ids = test_df['PassengerId']
    
    def separate_passenger_ids(df):
        df = df.copy()
        
        split_ids = df["PassengerId"].str.split("_", expand=True)
        
        df["GroupId"] = split_ids[0].astype(int)
        df["GroupMemberId"] = split_ids[1].astype(int)
        
        return df
    
    def fill_home_planet_by_group(df):
        group_planet_map = df.dropna(subset=['HomePlanet']).set_index('GroupId')['HomePlanet'].to_dict()
        
        df['HomePlanet'] = df['HomePlanet'].fillna(df['GroupId'].map(group_planet_map))
        
        return df
    
    def split_cabin(df):
        df['Cabin'] = df.groupby('GroupId')['Cabin'].transform(lambda x: x.ffill().bfill())
        cabin_split = df['Cabin'].str.split('/', expand=True)
        
        df['CabinDeck'] = cabin_split[0]
        df['CabinNum'] = cabin_split[1].astype(float)
        df['CabinSide'] = cabin_split[2]
        
        df['CabinDeck'] = df['CabinDeck'].fillna(df['CabinDeck'].mode()[0])
        df['CabinSide'] = df['CabinSide'].fillna(df['CabinSide'].mode()[0])
        df['CabinNum'] = df['CabinNum'].fillna(df['CabinNum'].median())
        
        return df

    def modify_dataframe(train, test):
        train = train.copy()
        test = test.copy()
        
        drop_cols = ['Name', 'Cabin'] 
        train.drop(columns=drop_cols, inplace=True, errors='ignore')
        test.drop(columns=drop_cols, inplace=True, errors='ignore')
    
        for col in train.columns:
            if col == 'Transported': continue # Don't encode the target
            
            if train[col].dtype == object or train[col].dtype == bool:
                
                train[col] = train[col].fillna('Unknown').astype(str)
                test[col] = test[col].fillna('Unknown').astype(str)
                
                le = LabelEncoder()
                le.fit(list(train[col]) + list(test[col])) 
                train[col] = le.transform(train[col])
                test[col] = le.transform(test[col])
            else:
                train[col] = train[col].fillna(0)
                test[col] = test[col].fillna(0)
                
        return train, test

    train_df = separate_passenger_ids(train_df)
    test_df = separate_passenger_ids(test_df)

    train_df = fill_home_planet_by_group(train_df)
    test_df = fill_home_planet_by_group(test_df)
    
    train_df = split_cabin(train_df)
    test_df = split_cabin(test_df)

    train_df, test_df = modify_dataframe(train_df, test_df)
    
    y = train_df['Transported'].values
    X = train_df.drop('Transported', axis=1).values
    
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X = scaler.fit_transform(X)
    X_sub = scaler.transform(test_df.values)
    
    return jnp.array(X), jnp.array(y).astype(float), jnp.array(X_sub), ids

X, y, X_submission, passenger_ids = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# --- 2. Model Kurulumu ---
num_features = X_train.shape[1]
n_wires = int(np.ceil(np.log2(num_features)))
n_layers = 5
dev = qml.device('lightning.qubit', wires=n_wires)

@qml.qnode(dev, interface="jax")
def circuit(data, weights):

    qml.AmplitudeEmbedding(data, wires=range(n_wires), normalize=True, pad_with=0)
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

circuit_vmap = jax.vmap(circuit, in_axes=(0, None))

def model(data, weights, bias):
    return circuit_vmap(data, weights) + bias

# --- 3. Loss ve Update ---
def loss_fn(params, data, targets):
    predictions = model(data, params["weights"], params["bias"])
    return optax.sigmoid_binary_cross_entropy(predictions, targets).mean()

@jax.jit
def update_step(params, opt_state, data, targets):
    loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

EPOCHS = 30

optimizer = optax.adamw(learning_rate=0.05, weight_decay=0.05)

shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
weights = np.random.normal(loc=0.0, scale=0.01, size=shape)

bias = jnp.array(0.0)
params = {"weights": weights, "bias": bias}
opt_state = optimizer.init(params)

# --- 4. TQDM ile Eğitim Döngüsü ---

BATCH_SIZE = len(X_train)

print("Eğitim Başlıyor...")

# TQDM'i burada kullanıyoruz
pbar = tqdm(range(EPOCHS), desc="Training")

for i in pbar:
    # 1. Adım: Güncelleme
    params, opt_state, loss_val = update_step(params, opt_state, X_train, y_train)
    
    pbar.set_postfix({
        "Loss": f"{loss_val.item():.4f}"
    })
    
print("Eğitim Tamamlandı.")
params_numpy = jax.tree_util.tree_map(lambda x: np.array(x), params)

np.savez(f"model_weights_{n_layers}layer.npz", **params_numpy)

print(f"Ağırlıklar 'model_weights_{n_layers}layer.npz' olarak kaydedildi.")

# --- 5. Değerlendirme (JIT Helper ile Hata Önleme) ---
@jax.jit
def predict(params, data):
    logits = model(data, params["weights"], params["bias"])
    return (logits > 0.0).astype(int)

val_acc = jnp.mean(predict(params, X_val) == y_val)
test_acc = jnp.mean(predict(params, X_test) == y_test)

print(f"\nValidation Accuracy: %{val_acc * 100:.2f}")
print(f"Test Accuracy: %{test_acc * 100:.2f}")

sub_preds = predict(params, X_submission)
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Transported": [True if x==1 else False for x in np.array(sub_preds)]
})
submission.to_csv(f'submission_{n_layers}layer.csv', index=False)