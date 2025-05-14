import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('classic')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
    titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

training_data = pd.read_csv('data/train.csv')

X = training_data.copy()
y = X.pop('SalePrice')

if 'id' in X.columns:
    X = X.drop(columns=['id'])

features_cat =[
    "MSZoning", 
    "Street", 
    "Alley", 
    "LotShape", 
    "LandContour", 
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "SaleType",
    "SaleCondition"
    ]

features_num = [
    "MSSubClass",
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold"
]

transformer_cat = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='NA'),
    OneHotEncoder(handle_unknown='ignore')
)

transformer_num = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75)
preprocessor = make_column_transformer(
    (transformer_cat, features_cat),
    (transformer_num, features_num),
    remainder='drop'
)

X_train_trans = preprocessor.fit_transform(X_train)
X_valid_trans = preprocessor.transform(X_valid)

input_shape = X_train_trans.shape[1]
model = keras.Sequential([
    layers.InputLayer(shape=(input_shape,)),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.05),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.05),
    layers.Dense(1, activation="linear"),
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train_trans, y_train,
    validation_data=(X_valid_trans, y_valid),
    batch_size=512,
    epochs=1000,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title='Loss')
history_df.loc[:, ['mean_absolute_error', 'val_mean_absolute_error']].plot(title='Mean Absolute Error')
plt.show()

# Apply model to test set for submission
test = pd.read_csv('data/test.csv')
ids = test['Id']

X_test = test.drop(columns=['Id'])
X_test_trans = preprocessor.transform(X_test)

y_pred_probs = model.predict(X_test_trans, batch_size=512)
y_pred = y_pred_probs.flatten()

submission = pd.DataFrame({
    'Id': ids,
    'SalePrice': y_pred
})
submission.to_csv('data/submission.csv', index=False)
print("Saved submission.csv with", len(submission), "rows")