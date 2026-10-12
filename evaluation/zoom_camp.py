{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3fb05700",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.set_autosave_interval(0)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Autosave disabled\n"
     ]
    }
   ],
   "source": [
    "%autosave 0"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "464b9b4a",
   "metadata": {},
   "source": [
    "# 4. Evaluation Metrics for Classification\n",
    "\n",
    "In the previous session we trained a model for predicting churn. How do we know if it's good?\n",
    "\n",
    "\n",
    "## 4.1 Evaluation metrics: session overview \n",
    "\n",
    "* Dataset: https://www.kaggle.com/blastchar/telco-customer-churn\n",
    "* https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv\n",
    "\n",
    "\n",
    "*Metric* - function that compares the predictions with the actual values and outputs a single number that tells how good the predictions are"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b3ff7c44",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/lib/python3.11/site-packages/pandas/core/computation/expressions.py:23: UserWarning: Pandas requires version '2.10.2' or newer of 'numexpr' (version '2.8.7' currently installed).\n",
      "  from pandas.core.computation.check import NUMEXPR_INSTALLED\n",
      "/opt/anaconda3/lib/python3.11/site-packages/pandas/core/arrays/masked.py:56: UserWarning: Pandas requires version '1.4.2' or newer of 'bottleneck' (version '1.3.7' currently installed).\n",
      "  from pandas.core import (\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ff66e768",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction import DictVectorizer\n",
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "81ee3ecc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Current working directory: /Users/foroghakbari/Zoomcamp/ml_course/zoomcamp_ml/evaluation\n",
      "\n",
      "Downloading from https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv\n",
      "\n",
      "File successfully downloaded to data/data-week-3.csv\n",
      "File size: 977501 bytes\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import urllib.request\n",
    "\n",
    "\n",
    "print(\"Current working directory:\", os.getcwd())\n",
    "\n",
    "# Create data directory if it doesn't exist\n",
    "if not os.path.exists('data'):\n",
    "    os.makedirs('data')\n",
    "    print(\"Created 'data' directory\")\n",
    "\n",
    "# Download the file using urllib instead of wget\n",
    "\n",
    "url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'\n",
    "output_path = 'data/data-week-3.csv'\n",
    "\n",
    "print(f\"\\nDownloading from {url}\")\n",
    "urllib.request.urlretrieve(url, output_path)\n",
    "\n",
    "# Verify the file was downloaded\n",
    "if os.path.exists(output_path):\n",
    "    print(f\"\\nFile successfully downloaded to {output_path}\")\n",
    "    print(\"File size:\", os.path.getsize(output_path), \"bytes\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4226455e-d145-4417-98a3-3af3701b5a79",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0        No\n",
       "1        No\n",
       "2       Yes\n",
       "3        No\n",
       "4       Yes\n",
       "       ... \n",
       "7038     No\n",
       "7039     No\n",
       "7040     No\n",
       "7041    Yes\n",
       "7042     No\n",
       "Name: churn, Length: 7043, dtype: str"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('data/data-week-3.csv')\n",
    "df.columns = df.columns.str.lower().str.replace(' ', '_')\n",
    "df.churn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "7beb2bfe-0bfa-4cea-98f2-3277993855d4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1]\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv('data/data-week-3.csv')\n",
    "df.columns = df.columns.str.lower().str.replace(' ', '_')\n",
    "\n",
    "categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)\n",
    "\n",
    "for c in categorical_columns:\n",
    "    df[c] = df[c].str.lower().str.replace(' ', '_')\n",
    "\n",
    "df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')\n",
    "df.totalcharges = df.totalcharges.fillna(0)\n",
    "\n",
    "df.churn = (df.churn == 'Yes').astype(int)\n",
    "print(np.unique(df.churn))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "b29a2af5-7b3a-4bcb-a019-c66e4a6bedc8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "customerid                   5442-PPTJY\n",
       "gender                             Male\n",
       "seniorcitizen                         0\n",
       "partner                             Yes\n",
       "dependents                          Yes\n",
       "tenure                               12\n",
       "phoneservice                        Yes\n",
       "multiplelines                        No\n",
       "internetservice                      No\n",
       "onlinesecurity      No internet service\n",
       "onlinebackup        No internet service\n",
       "deviceprotection    No internet service\n",
       "techsupport         No internet service\n",
       "streamingtv         No internet service\n",
       "streamingmovies     No internet service\n",
       "contract                       Two year\n",
       "paperlessbilling                     No\n",
       "paymentmethod              Mailed check\n",
       "monthlycharges                     19.7\n",
       "totalcharges                     258.35\n",
       "churn                                 0\n",
       "Name: 1814, dtype: object"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_full_train.iloc[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "0b1903b3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1]\n"
     ]
    }
   ],
   "source": [
    "df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)\n",
    "df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)\n",
    "\n",
    "df_train = df_train.reset_index(drop=True)\n",
    "df_val = df_val.reset_index(drop=True)\n",
    "df_test = df_test.reset_index(drop=True)\n",
    "\n",
    "import numpy as np\n",
    "\n",
    "y_train = df_train.churn.values\n",
    "\n",
    "print(np.unique(y_train))\n",
    "\n",
    "y_val = df_val.churn.values\n",
    "y_test = df_test.churn.values\n",
    "\n",
    "del df_train['churn']\n",
    "del df_val['churn']\n",
    "del df_test['churn']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "ab4132a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "numerical = ['tenure', 'monthlycharges', 'totalcharges']\n",
    "\n",
    "categorical = [\n",
    "    'gender',\n",
    "    'seniorcitizen',\n",
    "    'partner',\n",
    "    'dependents',\n",
    "    'phoneservice',\n",
    "    'multiplelines',\n",
    "    'internetservice',\n",
    "    'onlinesecurity',\n",
    "    'onlinebackup',\n",
    "    'deviceprotection',\n",
    "    'techsupport',\n",
    "    'streamingtv',\n",
    "    'streamingmovies',\n",
    "    'contract',\n",
    "    'paperlessbilling',\n",
    "    'paymentmethod',\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cc4583b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "dv = DictVectorizer(sparse=False)\n",
    "\n",
    "train_dict = df_train[categorical + numerical].to_dict(orient='records')\n",
    "X_train = dv.fit_transform(train_dict)\n",
    "\n",
    "model = LogisticRegression()\n",
    "#model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0fd1bd18-82a1-4729-ab31-d673e466ba6c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ab42db33-3cf0-4d03-9efd-2712fb24135d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "35c6ffee-8b66-4e71-bd27-e301e1b9c6a7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 1, ..., 1, 0, 1])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "4ea51f38",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8034066713981547"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "val_dict = df_val[categorical + numerical].to_dict(orient='records')\n",
    "X_val = dv.transform(val_dict)\n",
    "\n",
    "y_pred = model.predict_proba(X_val)[:, 1]\n",
    "churn_decision = (y_pred >= 0.5)\n",
    "(y_val == churn_decision).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3d8a7559",
   "metadata": {},
   "source": [
    "## 4.2 Accuracy and dummy model\n",
    "\n",
    "* Evaluate the model on different thresholds\n",
    "* Check the accuracy of dummy baselines"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "948eda88",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1409"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(y_val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "fc59dff6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8034066713981547"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(y_val == churn_decision).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "6bcb4def",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8034066713981547"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "1132/ 1409"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "adf1012b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "ffb75fc5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8034066713981547"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_val, y_pred >= 0.5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "27840301",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.00 0.274\n",
      "0.05 0.509\n",
      "0.10 0.591\n",
      "0.15 0.666\n",
      "0.20 0.710\n",
      "0.25 0.739\n",
      "0.30 0.760\n",
      "0.35 0.772\n",
      "0.40 0.785\n",
      "0.45 0.793\n",
      "0.50 0.803\n",
      "0.55 0.801\n",
      "0.60 0.795\n",
      "0.65 0.786\n",
      "0.70 0.765\n",
      "0.75 0.744\n",
      "0.80 0.735\n",
      "0.85 0.726\n",
      "0.90 0.726\n",
      "0.95 0.726\n",
      "1.00 0.726\n"
     ]
    }
   ],
   "source": [
    "thresholds = np.linspace(0, 1, 21)\n",
    "\n",
    "scores = []\n",
    "\n",
    "for t in thresholds:\n",
    "    score = accuracy_score(y_val, y_pred >= t)\n",
    "    print('%.2f %.3f' % (t, score))\n",
    "    scores.append(score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "11f492a8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x12640ea50>]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(thresholds, scores)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "174b048b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import Counter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "e0897055",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Counter({False: 1409})"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Counter(y_pred >= 1.0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "8b8ece7e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7260468417317246"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "1 - y_val.mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1600912",
   "metadata": {},
   "source": [
    "## 4.3 Confusion table\n",
    "\n",
    "* Different types of errors and correct decisions\n",
    "* Arranging them in a table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "501711e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "actual_positive = (y_val == 1)\n",
    "actual_negative = (y_val == 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "d898b9ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "t = 0.5\n",
    "predict_positive = (y_pred >= t)\n",
    "predict_negative = (y_pred < t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "ce46cfef",
   "metadata": {},
   "outputs": [],
   "source": [
    "tp = (predict_positive & actual_positive).sum()\n",
    "tn = (predict_negative & actual_negative).sum()\n",
    "\n",
    "fp = (predict_positive & actual_negative).sum()\n",
    "fn = (predict_negative & actual_positive).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "778f4ef6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[922, 101],\n",
       "       [176, 210]])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix = np.array([\n",
    "    [tn, fp],\n",
    "    [fn, tp]\n",
    "])\n",
    "confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "4a7aab64",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.65, 0.07],\n",
       "       [0.12, 0.15]])"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(confusion_matrix / confusion_matrix.sum()).round(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a37b1d4c",
   "metadata": {},
   "source": [
    "## 4.4 Precision and Recall"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "7dfcf399",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6752411575562701"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "p = tp / (tp + fp)\n",
    "p "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "4e180eec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5440414507772021"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r = tp / (tp + fn)\n",
    "r"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5088b30e",
   "metadata": {},
   "source": [
    "## 4.5 ROC Curves\n",
    "\n",
    "### TPR and FRP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "f8b9350d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5440414507772021"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tpr = tp / (tp + fn)\n",
    "tpr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "56d84d27-9b8e-45a9-a3d1-d5db9f402b91",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5440414507772021"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "0a8b418e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.09872922776148582"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fpr = fp / (fp + tn)\n",
    "fpr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "fc8e1021",
   "metadata": {},
   "outputs": [],
   "source": [
    "#for multiple thresholds \n",
    "\n",
    "scores = []\n",
    "\n",
    "thresholds = np.linspace(0, 1, 101)\n",
    "\n",
    "for t in thresholds:\n",
    "    actual_positive = (y_val == 1)\n",
    "    actual_negative = (y_val == 0)\n",
    "    \n",
    "    predict_positive = (y_pred >= t)\n",
    "    predict_negative = (y_pred < t)\n",
    "\n",
    "    tp = (predict_positive & actual_positive).sum()\n",
    "    tn = (predict_negative & actual_negative).sum()\n",
    "\n",
    "    fp = (predict_positive & actual_negative).sum()\n",
    "    fn = (predict_negative & actual_positive).sum()\n",
    "    \n",
    "    scores.append((t, tp, fp, fn, tn))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "76beba9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "columns = ['threshold', 'tp', 'fp', 'fn', 'tn']\n",
    "df_scores = pd.DataFrame(scores, columns=columns)\n",
    "\n",
    "df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)\n",
    "df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "e125f2fe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x126c6ff50>"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')\n",
    "plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f09dc644",
   "metadata": {},
   "source": [
    "### Random model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "b3a4b466",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.random.seed(1) #every customer randomly assigned\n",
    "y_rand = np.random.uniform(0, 1, size=len(y_val))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "76b5eff5-f907-4020-b124-c7394106d1e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([4.17022005e-01, 7.20324493e-01, 1.14374817e-04, ...,\n",
       "       7.73916250e-01, 3.34276405e-01, 8.89982208e-02])"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_rand"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "c6c24296",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5017743080198722"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "((y_rand >= 0.5) == y_val).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "ba13786f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def tpr_fpr_dataframe(y_val, y_pred):\n",
    "    scores = []\n",
    "\n",
    "    thresholds = np.linspace(0, 1, 101)\n",
    "\n",
    "    for t in thresholds:\n",
    "        actual_positive = (y_val == 1)\n",
    "        actual_negative = (y_val == 0)\n",
    "\n",
    "        predict_positive = (y_pred >= t)\n",
    "        predict_negative = (y_pred < t)\n",
    "\n",
    "        tp = (predict_positive & actual_positive).sum()\n",
    "        tn = (predict_negative & actual_negative).sum()\n",
    "\n",
    "        fp = (predict_positive & actual_negative).sum()\n",
    "        fn = (predict_negative & actual_positive).sum()\n",
    "\n",
    "        scores.append((t, tp, fp, fn, tn))\n",
    "\n",
    "    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']\n",
    "    df_scores = pd.DataFrame(scores, columns=columns)\n",
    "\n",
    "    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)\n",
    "    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)\n",
    "    \n",
    "    return df_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "f7af4bd1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_rand = tpr_fpr_dataframe(y_val, y_rand)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "dab6acab-df85-4169-968d-1e7870ca9103",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>threshold</th>\n",
       "      <th>tp</th>\n",
       "      <th>fp</th>\n",
       "      <th>fn</th>\n",
       "      <th>tn</th>\n",
       "      <th>tpr</th>\n",
       "      <th>fpr</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.00</td>\n",
       "      <td>386</td>\n",
       "      <td>1023</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.01</td>\n",
       "      <td>382</td>\n",
       "      <td>1013</td>\n",
       "      <td>4</td>\n",
       "      <td>10</td>\n",
       "      <td>0.989637</td>\n",
       "      <td>0.990225</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.02</td>\n",
       "      <td>379</td>\n",
       "      <td>993</td>\n",
       "      <td>7</td>\n",
       "      <td>30</td>\n",
       "      <td>0.981865</td>\n",
       "      <td>0.970674</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.03</td>\n",
       "      <td>375</td>\n",
       "      <td>979</td>\n",
       "      <td>11</td>\n",
       "      <td>44</td>\n",
       "      <td>0.971503</td>\n",
       "      <td>0.956989</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.04</td>\n",
       "      <td>372</td>\n",
       "      <td>969</td>\n",
       "      <td>14</td>\n",
       "      <td>54</td>\n",
       "      <td>0.963731</td>\n",
       "      <td>0.947214</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>96</th>\n",
       "      <td>0.96</td>\n",
       "      <td>14</td>\n",
       "      <td>41</td>\n",
       "      <td>372</td>\n",
       "      <td>982</td>\n",
       "      <td>0.036269</td>\n",
       "      <td>0.040078</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>97</th>\n",
       "      <td>0.97</td>\n",
       "      <td>11</td>\n",
       "      <td>36</td>\n",
       "      <td>375</td>\n",
       "      <td>987</td>\n",
       "      <td>0.028497</td>\n",
       "      <td>0.035191</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>98</th>\n",
       "      <td>0.98</td>\n",
       "      <td>8</td>\n",
       "      <td>19</td>\n",
       "      <td>378</td>\n",
       "      <td>1004</td>\n",
       "      <td>0.020725</td>\n",
       "      <td>0.018573</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>99</th>\n",
       "      <td>0.99</td>\n",
       "      <td>6</td>\n",
       "      <td>7</td>\n",
       "      <td>380</td>\n",
       "      <td>1016</td>\n",
       "      <td>0.015544</td>\n",
       "      <td>0.006843</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>100</th>\n",
       "      <td>1.00</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>386</td>\n",
       "      <td>1023</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>101 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     threshold   tp    fp   fn    tn       tpr       fpr\n",
       "0         0.00  386  1023    0     0  1.000000  1.000000\n",
       "1         0.01  382  1013    4    10  0.989637  0.990225\n",
       "2         0.02  379   993    7    30  0.981865  0.970674\n",
       "3         0.03  375   979   11    44  0.971503  0.956989\n",
       "4         0.04  372   969   14    54  0.963731  0.947214\n",
       "..         ...  ...   ...  ...   ...       ...       ...\n",
       "96        0.96   14    41  372   982  0.036269  0.040078\n",
       "97        0.97   11    36  375   987  0.028497  0.035191\n",
       "98        0.98    8    19  378  1004  0.020725  0.018573\n",
       "99        0.99    6     7  380  1016  0.015544  0.006843\n",
       "100       1.00    0     0  386  1023  0.000000  0.000000\n",
       "\n",
       "[101 rows x 7 columns]"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_rand"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "6ea930c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x126c2e010>"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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Wvfzhx9+aJYqVSrVtrFjx5rFihXLsIzZze2UESu+C7PtaZqEhAS2bt1K27ZtU21v27Yt69evv+F7NmzYcN34du3asWXLFhITEzMsqzu7k/38T06nk5iYGPLly5cREbONO93X06ZN49ChQ4wcOTKjI2YLd7KfFy1aRO3atfnggw8oWrQo5cqVY9iwYcTGxmZGZLd0J/u5YcOGnDx5kiVLlmCaJmfPnmX+/Pncc889mRE5x7Diu9AtbpR3JyIjI0lOTiYwMDDV9sDAQM6cOXPD95w5c+aG45OSkoiMjKRIkSIZltdd3cl+/qePP/6Yq1ev0q1bt4yImG3cyb4+cOAAL730EmvWrMHhyLb/d09Xd7KfDx8+zNq1a/H29ub7778nMjKSp556iosXL2reyE3cyX5u2LAhM2fOpHv37sTFxZGUlMS9997LuHHjMiNyjmHFd2G2PTLyN8MwUj03TfO6bf82/kbbJbW07ue/zZ49m9dff505c+ZQqFChjIqXrdzuvk5OTqZnz5688cYblCtXLrPiZRtp+TvtdDoxDIOZM2dSt25d7r77bkaPHs306dN1dORfpGU/79mzh4EDB/Laa6+xdetWfvnlF44cOcKAAQMyI2qOktnfhdn2n0oFChTAbrdf17DPnTt3XeP7W+HChW843uFwkD9//gzL6s7uZD//bc6cOfTv35958+bRunXrjIyZLaR1X8fExLBlyxbCw8N55plnANeXpmmaOBwOli1bRsuWLTMluzu5k7/TRYoUoWjRoqlulV6xYkVM0+TkyZOULVs2QzO7ozvZz6NGjaJRo0Y8//zzAFSrVg0/Pz+aNGnC22+/raPX6cSK78Jse2TE09OT0NBQwsLCUm0PCwujYcOGN3xPgwYNrhu/bNkyateujYeHR4ZldWd3sp/BdUSkb9++zJo1S+d7b1Na97W/vz87d+5k+/btKY8BAwZQvnx5tm/fTr169TIrulu5k7/TjRo14vTp01y5ciVl2/79+7HZbBQrVixD87qrO9nP165dw2ZL/bVlt9uB//3LXf47S74LM2xqbBbw92VjU6ZMMffs2WMOHjzY9PPzM48ePWqapmm+9NJLZu/evVPG/30505AhQ8w9e/aYU6ZM0aW9tyGt+3nWrFmmw+EwJ0yYYEZERKQ8Ll++bNWv4DbSuq//SVfT3J607ueYmBizWLFiZpcuXczdu3ebq1atMsuWLWs++uijVv0KbiGt+3natGmmw+EwJ06caB46dMhcu3atWbt2bbNu3bpW/QpuISYmxgwPDzfDw8NNwBw9erQZHh6ecgl1VvguzNZlxDRNc8KECWaJEiVMT09Ps1atWuaqVatSXnv44YfNZs2apRq/cuVKs2bNmqanp6dZsmRJc9KkSZmc2D2lZT83a9bMBK57PPzww5kf3A2l9e/0/6cycvvSup/37t1rtm7d2vTx8TGLFStmDh061Lx27Vomp3Y/ad3PY8eONStVqmT6+PiYRYoUMXv16mWePHkyk1O7lxUrVtzyv7lZ4bvQME0d2xIRERHrZNs5IyIiIuIeVEZERETEUiojIiIiYimVEREREbGUyoiIiIhYSmVERERELKUyIiIiIpZSGRERERFLqYyIiIiIpVRGRERExFIqIyIiImIplRERERGx1P8B2Qg900dDxzMAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')\n",
    "plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1958bf4d",
   "metadata": {},
   "source": [
    "### Ideal model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "b56b21df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1023, 386)"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "num_neg = (y_val == 0).sum()\n",
    "num_pos = (y_val == 1).sum()\n",
    "num_neg, num_pos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "7f051585",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "y_ideal = np.repeat([0, 1], [num_neg, num_pos])\n",
    "y_ideal\n",
    "\n",
    "y_ideal_pred = np.linspace(0, 1, len(y_val))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "77657220-7ba6-4fa5-89d2-081424e26e73",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 1, 1, 1])"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_ideal"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "658a88ca-893d-4493-bb06-98fb76062fb4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.00000000e+00, 7.10227273e-04, 1.42045455e-03, ...,\n",
       "       9.98579545e-01, 9.99289773e-01, 1.00000000e+00])"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_ideal_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "f7f9c21f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7260468417317246"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "1 - y_val.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "c85adf63",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_ideal, y_ideal_pred >= 0.726)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "b30738fe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>threshold</th>\n",
       "      <th>tp</th>\n",
       "      <th>fp</th>\n",
       "      <th>fn</th>\n",
       "      <th>tn</th>\n",
       "      <th>tpr</th>\n",
       "      <th>fpr</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>386</td>\n",
       "      <td>1023</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>0.1</td>\n",
       "      <td>386</td>\n",
       "      <td>882</td>\n",
       "      <td>0</td>\n",
       "      <td>141</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.862170</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>0.2</td>\n",
       "      <td>386</td>\n",
       "      <td>741</td>\n",
       "      <td>0</td>\n",
       "      <td>282</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.724340</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>0.3</td>\n",
       "      <td>386</td>\n",
       "      <td>600</td>\n",
       "      <td>0</td>\n",
       "      <td>423</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.586510</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40</th>\n",
       "      <td>0.4</td>\n",
       "      <td>386</td>\n",
       "      <td>459</td>\n",
       "      <td>0</td>\n",
       "      <td>564</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.448680</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50</th>\n",
       "      <td>0.5</td>\n",
       "      <td>386</td>\n",
       "      <td>319</td>\n",
       "      <td>0</td>\n",
       "      <td>704</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.311828</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>60</th>\n",
       "      <td>0.6</td>\n",
       "      <td>386</td>\n",
       "      <td>178</td>\n",
       "      <td>0</td>\n",
       "      <td>845</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.173998</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>70</th>\n",
       "      <td>0.7</td>\n",
       "      <td>386</td>\n",
       "      <td>37</td>\n",
       "      <td>0</td>\n",
       "      <td>986</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.036168</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>80</th>\n",
       "      <td>0.8</td>\n",
       "      <td>282</td>\n",
       "      <td>0</td>\n",
       "      <td>104</td>\n",
       "      <td>1023</td>\n",
       "      <td>0.730570</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>90</th>\n",
       "      <td>0.9</td>\n",
       "      <td>141</td>\n",
       "      <td>0</td>\n",
       "      <td>245</td>\n",
       "      <td>1023</td>\n",
       "      <td>0.365285</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>100</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>385</td>\n",
       "      <td>1023</td>\n",
       "      <td>0.002591</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     threshold   tp    fp   fn    tn       tpr       fpr\n",
       "0          0.0  386  1023    0     0  1.000000  1.000000\n",
       "10         0.1  386   882    0   141  1.000000  0.862170\n",
       "20         0.2  386   741    0   282  1.000000  0.724340\n",
       "30         0.3  386   600    0   423  1.000000  0.586510\n",
       "40         0.4  386   459    0   564  1.000000  0.448680\n",
       "50         0.5  386   319    0   704  1.000000  0.311828\n",
       "60         0.6  386   178    0   845  1.000000  0.173998\n",
       "70         0.7  386    37    0   986  1.000000  0.036168\n",
       "80         0.8  282     0  104  1023  0.730570  0.000000\n",
       "90         0.9  141     0  245  1023  0.365285  0.000000\n",
       "100        1.0    1     0  385  1023  0.002591  0.000000"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)\n",
    "df_ideal[::10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "60cb9f81",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x126cbb290>"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR')\n",
    "plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR')\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "434d98c9",
   "metadata": {},
   "source": [
    "### Putting everything together"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "d7a28486",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x133045ed0>"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiMAAAGdCAYAAADAAnMpAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAACLYklEQVR4nOzddVzV9xfH8de9XDoNJBS7dRZ2J06domLMzk2nmzXbpdOpmzo3nW6zWwzsmMzuAmZgd4CYoCJ9f398hcnPAgS+917O8/G4D75cLtw3lzkOnzgfjV6v1yOEEEIIoRKt2gGEEEIIkbVJMSKEEEIIVUkxIoQQQghVSTEihBBCCFVJMSKEEEIIVUkxIoQQQghVSTEihBBCCFVJMSKEEEIIVenUDpASCQkJ3LlzB3t7ezQajdpxhBBCCJECer2eJ0+e4O7ujlb75vEPoyhG7ty5g4eHh9oxhBBCCJEGN2/eJE+ePG/8uFEUI/b29oDyzTg4OKicRgghhBApERERgYeHR9Lv8TcximIkcWrGwcFBihEhhBDCyLxriYUsYBVCCCGEqqQYEUIIIYSqpBgRQgghhKqkGBFCCCGEqqQYEUIIIYSqpBgRQgghhKqkGBFCCCGEqqQYEUIIIYSqpBgRQgghhKpSXYzs3buX5s2b4+7ujkajYd26de/8nD179uDp6YmVlRUFCxbkjz/+SEtWIYQQQpigVBcjz549o2zZssyYMSNFj7969SpNmzalVq1aBAYGMnr0aAYMGMCaNWtSHVYIIYQQpifVZ9M0adKEJk2apPjxf/zxB3nz5mXatGkAlChRguPHjzN58mR8fHxS+/RCCCGEMDEZflDeoUOH8PLySnZf48aNmTt3LrGxsZibm7/yOdHR0URHRye9HxERkSHZBgyeSwfdPKbqvsC1qBvZs2dHq5VlNEKIrKu0uyOtK+R+58FmQqSnDC9GQkNDcXFxSXafi4sLcXFx3L9/Hzc3t1c+Z8KECXz//fcZGywhgX6Wv1Dc6iajE36m4/nRPOdpxj6nEEIYgVO3w/m2eUkpSESmyfBiBF49Oliv17/2/kSjRo1iyJAhSe9HRETg4eGRvqG0WibeH8tUl4GUN7/E789H83FQJWL1yUdGNBoNOXLmxNXVBVdXV1xcXHFxccHKyjJ98wghhMoeR8ay9MgNFhy8RnRcAuNblkarlYJEZLwML0ZcXV0JDQ1Ndl9YWBg6nY4cOXK89nMsLS2xtMz4X/ZVK7am6TgPdvdoQf1soVzuEcs2m9YEBv1LYGAgAQEBhIWF8Qi49H+fW6FCBXr06EGnTp3Ili1bhmcVQojMUM7DiRFrTrL86A2iY+P5qU0ZdGYyfS0yVob/F1atWjX8/f2T3bd9+3YqVqz42vUimSl/fjhyuxKDDy8GrQ77a9toa3eCH8ePZ+vWrYSGhnL79m02btzI2LFjadWqFfny5QMgICCAL774Ajc3Nzp27MiOHTtISEhQ9fsRQoj31baiB9M+Lo+ZVoNf4G0GrggiNl7+3yYylkafOGeSQk+fPuXSJWWcoHz58kydOpV69eqRPXt28ubNy6hRo7h9+zaLFi0ClK29pUuXpk+fPnzyySccOnSIvn37snz58hTvpomIiMDR0ZHw8HAcHBxS+S2+2blzUKIEODjA432r0fj1Uj5QbwzUGf7GzwsLC2PFihXMnTuXkydPJt1foEABevToQffu3dN/WkkIITLRttOhfLE8gNh4PZ/XK8zQxsXUjiSMUIp/f+tTadeuXXrglVu3bt30er1e361bN32dOnWSfc7u3bv15cuX11tYWOjz58+vnzVrVqqeMzw8XA/ow8PDUxv3rSIj9XpQbg8f6vX6Q7P0+m8dlNv+X9/5+QkJCfrjx4/rP/vsM72jo2PSa6HRaPSNGzfWr1y5Uh8VFZWumYUQIrOsPn5Tn2/EJn3Fcf762Lh4teMII5TS39+pHhlRQ0aNjAC4usLduxAQAOXLA3t+hl3jlA82ngDV+qXo60RGRuLn58fcuXPZvXt30v05cuSgS5cu9OzZkw8++CBdswshREaKiUugyo//8Cgylvk9KlGvWC61Iwkjk9Lf31l+VVL+/Mrba9de3FFnGNR+MUXz9yg4OjtFX8fGxobOnTuza9cuLl68yOjRo3F3d+fBgwdMmzaNMmXKUKVKFYKCgtL5OxBCiIxhodPiXS43AGtO3FI5jTBlUozkV94mFSMA9UZDzcHK9ZahcHxeqr5m4cKFGT9+PNevX2fz5s20bt0anU7H0aNHqVevHkeOHEmP6EIIkeF8KuQBYHvwXcKfx6qcRpgqKUbyK2+TFSMaDTT4Fqp9rry/aTAELE7119bpdDRt2pQ1a9Zw69YtatSowePHj2nUqBH79+9/3+hCCJHhSud2oEguO2LiEthyKkTtOMJESTGSX3mbrBgBpSDxGgdVPlPe3/AFBC1P8/O4uLiwbds26tWrx5MnT2jcuDE7duxI89cTQojMoNFo8PFURkdkqkZklCxfjLxoG/JqMQJKQfLhBKjUG9DD+n5wclWan8vOzo7NmzfTuHFjIiMjadasGQsWLODOnTsYwTpiIUQW1ap8brQaOH79EdfuP1M7jjBBWb4YeXlk5LX1gEYDTX6GCt1AnwBrP4XTfml+Pmtra9avX4+3tzfR0dH06NGD3Llz4+rqSpMmTfjqq69kkasQwqC4OFhRs4gzAH6Bt1VOI0xRli9GEkdGIiLg8eM3PEirhY+mQbnOSkGypjcEb0jzc1paWrJq1SpGjBhBqVKl0Gq1hIWFsW3bNsaPH0/58uXx9PRk5syZPH5jKCGEyDw+FZRdNX4Bt0hIkJFckb6yfDFiYwO5Xmydf+1UTSKtFlr8BmU+Bn08rO4B57ak+XnNzc2ZOHEip0+f5smTJxw+fJhZs2bRpk0bzM3NCQgIoH///ri5udG7d29CQmThmBBCPV4lXbGz1HHr0XOOXnuodhxhYrJ8MQJvWcT6/7Rm0HImlG4DCXGwsitc2P7ez29jY0OVKlXo27cvq1at4s6dO/zyyy+ULl2aqKgo5s6dS9GiRZk0aRLR0dHv/XxCCJFa1hZmNPvADVBGR4RIT1KMkIpiBJSCpNWfULIlJMSCb2e4lL67YnLmzMmgQYM4efIk+/bto0qVKjx9+pSRI0dSqlQp1q1bR1xcXLo+pxBCvEvirpotp0J5HhOvchphSnRqBzAEqSpGAMx04DNHGR05twlWdISOvlCwbrrm0mg01KxZk4MHD7J06VJGjBjB5cuXadWqFVZWVnzwwQdUqFCB8uXL4+HhgUajSfpcOzs7qlevjpmZWbpmEkJkXZXyZyNvdhtuPIzk7zOhtCyfW+1IwkRk+bNpAGbNgn79oEULWL8+FZ8YF/NiqmYr6Kyh82rIXzPd8yV6+vQpP/74I7///jsRERHvfHxiwzUrK6sMyySEyFp+8b/ArzsuUqtIThb3qqJ2HGHgUvr7W4oRYOtWaNoUypSBf/9N5SfHRcOKTnDJH8xtofMayFct3TO+LCEhgcuXLxMYGEhAQACBgYHcv38/2WOCg4OJioqiQYMGrF+/Hltb2wzNJITIGm48iKT2z7vQauDgyAa4OsofO+LNpBhJhbNnoWRJcHCA8PA0fIHYKFj+MVzZBRZ20GUdeFRK75ipsmfPHpo1a8azZ8+oVasWmzZtypDXTgiR9bT74xBHrz1kxIfF+axuIbXjCAMmp/amQop6jbyNuRV8vAzy14KYp7CkNdw+kZ4RU61OnTr4+/vj4ODAvn378PLykp4lQoh00fpFz5E1Abeke7RIF1KMkIpeI29jYaMsYs1XA6IjYHEruBOUTgnTplq1auzcuZPs2bNz5MgR6tWr98p0jhBCpFbTMm5Y6rRcCnvKqdtpGU4WIjkpRl5I9Y6a17GwVQoSjyoQFQ6LvCH0VDqkSztPT0927dqFs7MzQUFB1K1bl9DQUFUzCSGMm4OVOY1LuQJyeJ5IH1KMvJAuxQiApT10Wg25K0LUY6UguRv8nl/0/ZQpU4a9e/fi5ubGmTNnqFOnDrduyf9AhBBplzhVs+HfO8TEJaicRhg7KUZeSLdiBMDKQdlV414eIh/AohZw73w6fOG0K168OHv37iVv3rxcuHCB2rVrcy1dvlkhRFZUq4gzuewteRQZy67zYWrHEUZOipEX0rUYAbB2gs5+4PoBPLsHC5vD/Yvp9MXTpnDhwuzdu5dChQpx9epVateuzblz51TNJIQwTmZaDa1eND2TqRrxvqQYeSHdixEAm+zQdQPkKgVP7yoFyYPL6fgEqZcvXz727NlD8eLFuXnzJpUqVWL16tWqZhJCGKfWFZT28LvOh/HwWYzKaYQxk2LkhQwpRuBFQbIenIvDkxBY2AIepfeTpE7u3LnZs2cPderU4enTp7Rt25Yvv/yS2NhYVXMJIYxLMVd7Sud2IDZez4ag22rHEUZMipEXEnuNhIensdfI29g5Q7eNkLMoRNyCBc3h8Y10fpLUyZUrF//88w/Dhw8HYOrUqdSvX587d+6omksIYVxal1dGR9b/K//vEGknxcgL6dJr5G3scilTNtkLQfgNZcomXN2/JHQ6HZMmTcLPzw8HBwf2799P+fLl2b17t6q5hBDG46Oybmg1EHjjMTcfRqodRxgpKUZekjg6kmGbTBzclBGSbPmVqZqFH0FESAY9Wcq1atWK48ePU6ZMGcLCwmjQoAGTJk2SzopCiHfKZW9F1YI5ANh4UkZHRNpIMfKSDFs38jLH3EpB4pgXHl5RRkieqr8trkiRIhw6dIiuXbuSkJDAyJEjadWqlbSQF0K8U4uy7gBs/Ff9P66EcZJi5CWZUowAOOWF7hvBIQ88uKgUJM/Ub9NuY2PDggUL+PPPP7GwsGD9+vV4enoyefJkduzYwcOHD9WOKIQwQB+WdkWn1XA2JIJLYU/UjiOMkBQjL8m0YgSUqZruG8HeHe6dUzq1Rqr/y16j0fDpp59y8OBB8ufPz5UrVxg2bBgNGzYkR44c5M+fny5durBjxw4SEqTrohACnGwsqF3UGYANMjoi0kCKkZdkajECkL2gMmVj5wp3TysFyfNHmfTkb+fp6cmJEyeYOHEiPj4+FCxYEIDr16+zZMkSGjZsSOHChfnhhx+4efOmymmFEGr7b6rmjqw3E6mm0RvBfzURERE4OjoSHh6Og4NDhj1PcDCUKgX29soWX40mw54quXsXYEFTpVOre3nosk7p4GpgHj9+TEBAAKtXr2bZsmWEhyundWq1Wn7++WeGDBmickIhhFqeRsfh+YM/0XEJbPqiJqVzO6odSRiAlP7+lpGRlxQqBGZm8OQJZGq7DeeiygiJTQ64EwhLfCAqIhMDpIyTkxP169dn5syZ3Llzh8WLF1O3bl0SEhL48ssv8fX1VTuiEEIldpY6GpZwAZTD84RIDSlGXmJpCYULK9fBmX3Qbq4SSh8S62xw+zgsbQPRhrsQzMbGhs6dO7Nr1y4GDhwIQLdu3Thw4IDKyYQQamle1g2ATf/eISHB4AfdhQGRYuT/lCypvM30YgTAtbQyRWPlCDePwNJ2EPNMhSCpM2XKFLy9vYmOjsbb25tLly6pHUkIoYK6xXJhZ6njTngUJ24Yxvo3YRykGPk/pUopb8+cUSmAeznoshYsHeDGQVjWHmIMu6uhmZkZS5cupWLFijx48ICmTZvy4MEDtWMJITKZlbkZXqWUqZqNMlUjUkGKkf+j6shIotye0NkPLOzh2j5Y0RFio1QM9G62trZs3LiRvHnzcvHiRVq2bMmzZ4Y/qiOESF/NX+yq2XIqhLh42f4vUkaKkf/zcjGi6j4jj0rQeTWY28KVXeDbCeKiVQz0bq6urmzZsiXpnJvGjRsTEWF4C3GFEBmnZuGcZLe14P7TGPZdUr+ZozAOUoz8n2LFQKuFR48gNFTlMHmrQqeVoLOGS//Ayq4QF6NyqLcrVaoU27dvx9HRkQMHDtCoUSMePZK5YyGyCnMzbVLPEb8AdQ8DFcZDipH/Y2WlbPEFladqEuWvCR19QWcFF7bB6h4QH6t2qreqUqUKO3fuJEeOHBw9epT69etz7949tWMJITJJ6wq5Adh+JpSIKMP+/5UwDFKMvIZBrBt5WcE68PEyMLOAc5tgTW+Ij1M71VtVqFCB3bt3kytXLoKCgqhbt650ahUii/ggtyNFctkRHZfA5pPSHl68mxQjr6H6jprXKdwA2i8FrTkEr4O1fSAhXu1Ub1W6dGn27NmDu7s7wcHBFC9enPHjxxMVZdiLcYUQ70ej0eDjmQcAv4BbKqcRxkCKkdcwuJGRREW9oN0i0Org9GpY18/gC5LixYuzb98+qlWrRmRkJF999RUlS5Zk7dq1cn6FECasVfncaDVw7Nojrj+QnXXi7aQYeY3EYuTMGZV31LxO8abQZh5ozODkCtgwAAz89NyCBQty4MABli5dSu7cubl69SqtW7emWbNmMkoihIlycbCiRuGcAKyRhaziHaQYeY3ixZVD8h4+hLAwtdO8Rklv8JkDGi0ELYFNgwy+INFoNHTs2JHz58/z9ddfY2VlxdatW/nyyy/VjiaEyCBtXpqqkfbw4m2kGHkNa2soWFC5NripmkSlW0Orv5SCJGAhbB1mgMM4r7K1tWXs2LGsW7cOgJkzZ7JmzRp1QwkhMoRXSVfsLHXcevScY9ceqh1HGDApRt7AYNeNvKxMW/CeCWjg2BzYNtIoChKAxo0bM2LECAB69erF1atXVU4khEhv1hZmNPtAOTxvjSxkFW8hxcgbGOSOmtcp1wFaTFeuj/wB278ymoLkhx9+oFq1aoSHh/Pxxx8TE2PYDd2EEKmX2HNky6lQnscY9oJ7oR4pRt7AKEZGElXoAh/9olwfmgE7xhpFQWJubs7y5ctxcnLi6NGjjBkzRu1IQoh0Vil/djyyW/M0Oo7twWq3tRaGSoqRNzCqYgSgYk9oOlm53j8Vdk9QN08K5cuXj/nz5wMwefJk/vrrLxIMfDGuECLltFoNrcsrC1lXn5CpGvF6Uoy8QYkSyo6ae/eUm1Go/Ak0flGE7JkE+6aqmyeFWrZsyRdffAFAnz59qFGjBkePHlU5lRAivfhUUIqRA5fuExou2/nFq6QYeQMbG8ifX7k2mtERgGr9oNFY5XrH93Biobp5Umjq1KlMmDABOzs7Dh8+TJUqVejevTshIdJKWghjlzeHDZXyZyNBD+uCpOeIeJUUI29hdFM1iWoMhJqDletNg+DsJlXjpIROp2PkyJGcP3+erl27ArBw4UKKFy+etA1YCGG8EkdH1py4Jd2XxSukGHkLo9lR8zoNvoXyXUCfAKt7wrX9aidKEXd3dxYuXMjhw4epWLEiERERtGrVihEjRhAXZ9iHAwoh3qxpGTcsdVouhj3l1O1wteMIAyPFyFsY7cgIKAtePpoGxT+C+GhY3gFC/lU7VYpVqVKFgwcPMniwMsLz008/0bBhQ0JDZTW+EMbIwcocr1KugDI6IsTLpBh5i8SREaMsRgDMdOAzF/LVhOgIWOID986rnSrFzM3NmTp1KqtWrcLe3p49e/ZQoUIFjh07pnY0IUQa+LzoObLh3zvExMmuOfEfKUbeonhx5e3du/DggbpZ0szcCjosA9cy8OweLGwO9y+pnSpV2rRpw7FjxyhVqhQhISG0bNmSe0azxUkIkahm4ZzksrfkUWQsu84b4sFfQi1SjLyFnR3ky6dcG+3oCICVI3RdD7lKwdO7SkHy8IraqVKlWLFiHDp0iOLFi3Pnzh26desm/UiEMDI6My0tyyujIzJVI14mxcg7JE7VnD6tbo73ZpNdKUici8OTO7CgOTy6pnaqVLG3t8fX1zfpxN+pU42jj4oQ4j+Ju2p2nQ/j4TM5AkIo0lSMzJw5kwIFCmBlZYWnpyf79u176+OXLl1K2bJlsbGxwc3NjR49evDASOY9ypZV3p44oW6OdGHnDF03QI4iEHFLGSF5fFPtVKlSpkwZpk2bBsCoUaM4fPiwuoGEEKlSzNWe0rkdiI3Xs/HfO2rHEQYi1cWIr68vgwYNYsyYMQQGBlKrVi2aNGnCjRs3Xvv4/fv307VrV3r16sWZM2dYtWoVx44do3fv3u8dPjNUqaK8PXJE3Rzpxt4Fum2E7IXg8Q1Y+BGEG1cTok8//ZS2bdsSFxfHxx9/zKNHj9SOJIRIhaSeI3KSr3gh1cXI1KlT6dWrF71796ZEiRJMmzYNDw8PZs2a9drHHz58mPz58zNgwAAKFChAzZo16dOnD8ePH3/v8JkhsRg5cwaePFE3S7pxcFMKkmz5lamahc0hwng6nWo0GmbPnk2BAgW4fv06vXv3lvUjQhiRFmXd0Wk1nLwVzsW7pvI/VvE+UlWMxMTEcOLECby8vJLd7+XlxcGDB1/7OdWrV+fWrVts2bIFvV7P3bt3Wb16Nc2aNXvj80RHRxMREZHsphZXV8ibVzkE10jqp5RxzA3dNoFTXnh4GRa1gKfGs7rd0dERX19fzM3N8fPz49NPPyU+Xo4nF8IY5LCzpG6xXACsCTCukVmRMVJVjNy/f5/4+HhcXFyS3e/i4vLGZlTVq1dn6dKltG/fHgsLC1xdXXFycmL69OlvfJ4JEybg6OiYdPPw8EhNzHRnclM1iZw8lBEShzxw/4IyQvLsvtqpUqxSpUrMmzcPrVbL3Llz6datm3RpFcJItPFUdtWsDbxFfIK0h8/q0rSAVaPRJHtfr9e/cl+i4OBgBgwYwDfffMOJEyfYtm0bV69epW/fvm/8+qNGjSI8PDzpdvOmuossTbYYAWWqptsGsHeDe+dgkTdEPlQ7VYp17tyZFStWoNPpWLp0KR06dCAmRlboC2Ho6hXPhaO1OXcjojlwyXj+CBIZI1XFSM6cOTEzM3tlFCQsLOyV0ZJEEyZMoEaNGgwbNowyZcrQuHFjZs6cybx58954IqulpSUODg7Jbmp6uRgxyfOdchRSpmzsXODuaWXKxogKkrZt27JmzRosLCxYvXo1bdq0UXVqTwjxbpY6M1qUdQfATxayZnmpKkYsLCzw9PTE398/2f3+/v5Ur179tZ8TGRmJVpv8aczMzACM5uTGChXAzAxCQuCWqf6byVlYmbKxdYbQU7CkNTx/rHaqFGvRogUbNmzAysqKjRs34uTkRLFixejQoQM//fQTJ0+eVDuiEOL/+Hgqu2q2nQnlSVSsymmEmlI9TTNkyBDmzJnDvHnzOHv2LIMHD+bGjRtJ0y6jRo1KOgIeoHnz5vj5+TFr1iyuXLnCgQMHGDBgAJUrV8bd3T39vpMMZGMDH3ygXJvkVE0i52JKHxKbHHAnUDnLJsp4RhgaN27M1q1bKViwIHq9ngsXLrBixQpGjBhBhQoV3tkPRwiRucrmcaSgsy1RsQlsPSWHYGZlqS5G2rdvz7Rp0xg7dizlypVj7969bNmyhXwv+qaHhIQk6znSvXt3pk6dyowZMyhdujRt27alWLFi+Pn5pd93kQlMet3Iy1xKKp1arbPB7eOwtC1EP1U7VYrVrVuXy5cvc/fuXbZt28aECROoVq0a8fHx9OrVi+fPn6sdUQjxgkajkZ4jAgCN3gjmSiIiInB0dCQ8PFy19SPz50PPnlCrFuzdq0qEzHUnSFk7EhUO+WpAp1VgYat2qjR5/PgxJUuWJCQkhJEjRzJhwgS1IwkhXrjz+Dk1Ju1Er4d9w+vhkd1G7UgiHaX097ecTZNCiSMjJ05Altg96l4OuqwFSwe4fgCWfwwxkWqnShMnJ6ekpnw///wzAQEBKicSQiRyd7KmeqEcAPhJz5EsS4qRFCpeHBwcIDLSBA7NS6ncntB5DVjYwdW9sKIjxEapnSpNvL29adu2bdJ0TWysLJYTwlAkTtX4Bd4ymo0NIn1JMZJCWi1UqqRcm/y6kZd5VIZOq8HcFq7sAt/OEBetdqo0mT59OtmyZSMoKIgpU6aoHUcI8cKHpV2xtTDj+oNITlyXs6ayIilGUiFxquboUXVzZLp81aDTStBZwyV/WNkN4oyvsZiLi0vSib/fffcd58+fVzeQEAIAGwsdTT5wA2Qha1YlxUgqZJkdNa+TvyZ09AWdFVzYCqt7QLzxTXV06dKFxo0bEx0dTYMGDTh06JDakYQQQOsKSnv4Tf+GEBUr50xlNVKMpEJiMRIcDFmywWfBOvDxMjCzhHObYE1viDeu1bwajYY5c+ZQrFgxbt++Te3atZk+fbrMUwuhsqoFcpDbyZon0XH4B99VO47IZFKMpIKLC+TLZ4In+KZG4QbQfgmYWUDwOljbBxKM66+YPHnycOzYMdq2bUtcXBwDBgygY8eOPH1qPP1UhDA1Wq0maXREpmqyHilGUqlyZeVtlpyqSVTUC9otAq0OTq+Gdf2MriCxt7fH19eXX375BZ1Ox4oVK6hatSoPHjxQO5oQWVar8koxsvfCPcIijHPnnkgbKUZSKUuvG3lZsSbQZj5ozODkCtgwABIS1E6VKhqNhkGDBrF7927c3d05c+YMPXv2lCkbIVRS0NmOCnmdSNDDuiDpOZKVSDGSSonFyOHDJnqCb2qUbAE+c0CjhaAlsHmw0RUkADVq1GDz5s1YWlqyYcMGfvvtN7UjCZFlJR6et+bEbfnDIAuRYiSVKlYEKyu4exfOnlU7jQEo3Rpa/aUUJCcWwNZhRlmllStXLqn3yLBhwzieZRcFCaGuj8q4Y6HTcv7uE87cyYo7BbImKUZSycoKatZUrv391c1iMMq0Be+ZgAaOzYFto4yyIOnXrx+tWrUiNjaWjz/+mIgsuWVKCHU5WpvTqKQLIAtZsxIpRtKgUSPlrRQjLynXAVpMV66PzAL/r42uINFoNMydO5d8+fJx+fJlPv30UxkmFkIFPi921WwIukNsvPFN/YrUk2IkDRKLkd27Icb4GpFmnApd4KNflOuD02HnD0ZXkGTLlo0VK1ag0+nw9fVl8uTJUpAIkclqF3Emp50lD57FsPv8PbXjiEwgxUgalC0Lzs7w7JmykFW8pGJPaDpZud43BXZPVDdPGlStWpUff/wRgOHDh9O1a1eePXumciohsg6dmZaW5dwB8JOpmixBipE00GqhQQPlWqZqXqPyJ9B4gnK9ZyLs/VndPGkwdOhQfvrpJ8zMzFiyZAlVq1blwoULascSIstI3FWz42wYj57JELSpk2IkjRKnav75R90cBqtaP2g0VrneOQ72T1M1TmppNBqGDRvGjh07cHFx4fTp01SsWBFfX1+ZthEiE5Rwc6CkmwMx8QlsOnlH7Tgig0kxkkaJxcjRo/D4sapRDFeNgVD/a+X6n2/h4Ax186RBnTp1CAwMpFatWjx58oSPP/6YIkWK8OOPP3L7tjRlEiIjJbaHXx0g/9ZMnRQjaeThAcWKKT2+du1SO40Bqz0U6o5WrrePgSN/qpsnDdzc3Ni5cydjxozBzs6Oy5cvM2bMGPLmzctHH33E2rVriY01vhOMhTB03uVyY6bV8O/Nx1wKk7OjTJkUI++hYUPlrawbeYe6I6D2MOV663ClF4mR0el0jBs3jtDQUObPn0/NmjVJSEhg8+bNtG7dmjx58jBs2DDOnTundlQhTIazvSV1izoDspDV1Ekx8h6k30gq1BsDNQYp15u/VLq1GiFbW1u6d+/Ovn37OHfuHMOHD8fFxYWwsDAmT55MiRIlqFGjBnPnzpVTgIVIB60rKAtZ1wbeJj5B1muZKilG3kPdumBmBpcuwbVraqcxcBoNNPwOqn2uvL9xEAQuUTPReytWrBiTJk3i5s2brFu3jhYtWmBmZsbBgwfp3bs3rq6u9OrVi4MHD8qiVyHSqEGJXDhY6QgJj+LwFTlV21RJMfIeHB3/OzhPRkdSQKMBr3FQpS+gh/Wfw7++aqd6b+bm5nh7e7N+/Xpu3rzJxIkTKVq0KM+ePWPevHnUqFGDkiVLMnnyZO7evat2XCGMipW5Gc3LKj1H1pyQqRpTJcXIe5KpmlTSaODDiVCxF6CHdX3h1Gq1U6UbNzc3RowYwblz59i3bx/du3fHxsaGc+fOMWzYMPLkyUOrVq3Yvn272lGFMBqJPUe2ng7laXScymlERpBi5D0lFiM7dkB8vLpZjIZGo3RprdAV9Ang9ymcWad2qnSl0WioWbMm8+fPJyQkhL/++osqVaoQFxfHunXraNy4MWPHjpXpGyFSoLyHEwVy2vI8Np5tp0PVjiMygBQj76lyZbC3h4cPITBQ7TRGRKuFj36Fcp1AHw9resG5zWqnyhAODg588sknHD58mNOnT9O3b18Avv32W0aPHi0FiRDvoNFokg7Pk6ka0yTFyHsyN4d69ZTrtWvVzWJ0tFrlpN8y7SEhDlZ2g/Pb1E6VoUqVKsWsWbP45RflQMGJEycyePBgKUiEeIdWL3bVHLrygFuPIlVOI9KbFCPpoEsX5e0ffyiH54lU0JqB90wo7QMJsbCyC1w0/R77gwYNYubMmQD8+uuvfPbZZyQkyFHpQrxJbidrqhXMAcBa6chqcqQYSQetWkGhQspUzbx5aqcxQmY6aPUXlGgB8TGwoiNcNv22tp999hnz5s1Do9Hw559/UrNmTRYuXCgnBAvxBokLWf0Cb8tooomRYiQdmJnBl18q11OnQpws9k49Mx20mQfFmkF8NCzvAFf3qZ0qw/Xo0YOlS5dibm7OoUOH6N69O25ubvTp04eAgAC14wlhUJqUdsXa3Iyr958RcOOx2nFEOpJiJJ107w45cyrNz1abzk7VzGVmDm3nQ5HGEPcclrWD6wfVTpXhOnTowJUrVxg3bhwFCxbkyZMn/PXXX3h6ejJjhvEdLihERrG11NGktCsg7eFNjRQj6cTaGj5/0Vz0559BRhDTSGcJ7RZBoQYQGwlL28KNI2qnynB58uRhzJgxXLx4kZ07d9K6dWsAvvjiCyZPnqxyOiEMR+JUzcZ/7xAVK/0UTIUUI+mof3+lKAkIkJN834u5FXy8FArWhZinsMQHbh1XO1Wm0Gq11KtXj9WrVzNmzBgAhg0bxg8//CBz5EIA1QrmwN3RioioOHacDVM7jkgnUoyko5w5oUcP5frnn9XNYvTMreHj5ZC/FsQ8gcWt4XbWWUOh0WgYN24c48aNA+Cbb75hzJgxUpCILE+r1dCy/IueIzJVYzKkGElnQ4Yo7TO2bYOTJ9VOY+QsbKDDCshbDaLDYXErCPlX7VSZasyYMUyZMgWACRMm8O2336qcSAj1JU7V7Llwj3tPolVOI9KDFCPprFAh8PFRrmWqPx1Y2kGnVZCnMkQ9hkUtIfS02qky1ZAhQ5IWso4fP54jR0x/DY0Qb1PI2Y5yHk7EJ+hZHyQ9R0yBFCMZYNgw5e2yZRAcrG4Wk2BpD51XQ25PeP4QFnlD2Dm1U2Wq/v3707lzZxISEujVqxcxMTFqRxJCVUnt4aUBmkmQYiQDVKoE3t7KwXmJ/UfEe7JyhM5rwK0sRN6Hhc3h3gW1U2WqX375BWdnZ86cOcOECRPUjiOEqpqXdcfCTMvZkAiC70SoHUe8JylGMsjkycq5Ndu2wZYtaqcxEdbZoMs6cPkAnoUpBcmDy2qnyjQ5c+Zk+vTpgDJdc+bMGZUTCaEeJxsLGpTIBUjPEVMgxUgGKVwYBg5UrocMgdhYdfOYDJvs0HU95CoJT0OVguThVbVTZZp27drRvHlzYmNj6dWrF/Hx0mdBZF0+Lw7PWxd0h7h4OdvJmEkxkoG++gqcneH8eZg1S+00JsQ2B3TdADmLQcRtWNgCwrPGvLFGo2HmzJk4ODhw5MiRpJESIbKiOsWcyWFrwf2n0ey9eE/tOOI9SDGSgRwd4UWbCL77Dh48UDWOabFzhm4bIEdhCL+hbPuNfKh2qkyRJ08efn7RyGbMmDEsWbJE+o+ILMncTEuLcu6ALGQ1dlKMZLBevaBMGXj0SClIRDqyd1XWkNi7w/3zSuv4mKxx4m3v3r1p1KgRkZGRdOnShRo1anDs2DG1YwmR6RKnavyD7xIeKfPhxkqKkQxmZga//KJcz5oFsuYwnTl5QJe1YOUEt4+DbxeIM/1tr1qtlg0bNvDjjz9ia2vLoUOHqFy5Mj169CA0NFTteEJkmlLuDhR3tScmLoFNp+6oHUekkRQjmaB+/f+2+vbsCXFxaicyMbmKK43RzG3g8g5Y9xkkmP5iNisrK0aNGsWFCxfo2rUrAAsWLKBIkSL89NNPREdLZ0ph+jQaDa0Te46ckF01xkqKkUwyfbqyhuToUZg4Ue00JsijsnLar1YHp1fDlqFZ5uhkd3d3Fi5cmDQ68vTpU0aMGEGpUqXYsGGDrCcRJq9ludxoNRBw4zFX72eNqVpTI8VIJvHwUAoSgO+/h8BAdfOYpCKNoOUfgAaOz4Vto7JMQQJQtWpVDh06xMKFC3F1deXy5ct4e3vz4YcfEiytgIUJy+VgRe2izoD0HDFWUoxkos6doXVrZZqmSxeIilI7kQkq0xZa/KZcH5kF/l9nqYJEq9XStWtXLly4wMiRI7GwsGD79u2UKVOGgQMH8ujRI7UjCpEhEhey+gXcJiEh6/ybNxVSjGQijQb++ANy5VIWsn79tdqJTFSFrvDRi1XDB6fDjrFZqiABsLe3Z8KECQQHB9OyZUvi4+P57bffKFKkCLNmzSJOFi4JE9OopAv2VjpuP37O4avSR8HYSDGSyZydYfZs5XrKFNi3T908JqtiT2j64tjk/VNhd9Y8y6VQoUKsXbsWf39/SpUqxYMHD+jXrx+FChXiu+++4/r162pHFCJdWJmb8VEZNwDWnJCeI8ZGihEVtGgBPXoof6x37w5yAGsGqfwJNH5RhOyZBHt+VjePiho2bEhQUBAzZswgR44c3Lhxg++//54CBQrg5eWFn5+fLHQVRi9xqmbr6RAiY2T0z5hIMaKSadPA1RWuXIHFi9VOY8Kq9YNGY5XrXeNg/y/q5lGRTqejf//+3Lx5k6VLl1K/fn30ej3+/v74+Pjw66+/qh1RiPfimS8b+XPYEBkTz7bT0m/HmEgxohIHBxg2TLn+8UfpPZKhagyE+i8W6PzzHRycoWoctVlbW9OxY0d27NjB5cuX6d+/PwDDhw/n+PHjKqcTIu2UniPK6Mga2VVjVKQYUVGfPsoakitXYNkytdOYuNpDoe4o5Xr7GDj8h7p5DETBggWZPn06rVq1IjY2lvbt2xMeHq52LCHSrFV5pQHawcsPuPP4ucppREpJMaIiW1v48kvlevx4pUOryEB1RkCtocr1thFwbI66eQyERqNh7ty55MuXjytXrtCnTx9ZPyKMlkd2G6oUyI5eD2sDZSGrsZBiRGX9+kH27HDhAqxcqXYaE6fRQP2vlGkbgM1fwokFqkYyFNmyZWPFihXodDp8fX2ZM0cKNWG8fDz/m6qRwto4pKkYmTlzJgUKFMDKygpPT0/2vWN/anR0NGPGjCFfvnxYWlpSqFAh5s2bl6bApsbeHgYPVq7HjcsSR6qoS6OBht9DVWWdBBsHQeBSVSMZiqpVqzJ+/HgABgwYwKlTp1ROJETaNP3ADStzLVfuPSPo5mO144gUSHUx4uvry6BBgxgzZgyBgYHUqlWLJk2acOPGjTd+Trt27dixYwdz587l/PnzLF++nOLFi79XcFPyxRfKuTXBweDnp3aaLECjgcbjoXIfQA/r+8O/vmqnMghDhw7lww8/JCoqihYtWnD16lW1IwmRanaWOj4s5QooHVmF4dPoUzmGVaVKFSpUqMCsWbOS7itRogQtW7ZkwoRXG0tt27aNjz/+mCtXrpA9e/Y0hYyIiMDR0ZHw8HAcHBzS9DUM3bffwtixUKaMcm6NVibQMp5eD5uHwPF5oNFC69nwQRu1U6nu3r17VK9enUuXLpEnTx527NhB0aJF1Y4lRKrsu3iPLnOP4mhtztExDbDUmakdKUtK6e/vVP3Ki4mJ4cSJE3h5eSW738vLi4MHD772czZs2EDFihX56aefyJ07N0WLFmXo0KE8f/7mVc7R0dFEREQku5m6gQOVKZuTJ2HjRrXTZBEaDTSdorSP1yeA36dwZp3aqVTn7OzMnj17KF68OLdu3aJOnTpy0J4wOtUL5cTVwYrw57HsPBumdhzxDqkqRu7fv098fDwuLi7J7ndxcSE09PUNZq5cucL+/fs5ffo0a9euZdq0aaxevTqpt8HrTJgwAUdHx6Sbh4dHamIapezZIfElkd5TmUirhY9+hbIdQR8Pa3rBuc1qp1Kdu7s7e/bsoUyZMoSGhlKnTh2CgoLUjiVEiplpNbR8sc13jUzVGLw0TQZoNJpk7+v1+lfuS5SQkIBGo2Hp0qVUrlyZpk2bMnXqVBYsWPDG0ZFRo0YRHh6edLt582ZaYhqdzz5Tfjfu2gVnz6qdJgvRasF7BnzQDhLiYGU3OL9N7VSqy5UrF7t27cLT05P79+9Tr149jh07pnYsIVKsjadSjOw+H8aDp9EqpxFvk6piJGfOnJiZmb0yChIWFvbKaEkiNzc3cufOjaOjY9J9JUqUQK/Xc+vW6zvkWVpa4uDgkOyWFeTNq5xbAzBzprpZshytGbScBaVaQ0IsrOwCF/9RO5XqsmfPzo4dO6hevTqPHz+mQYMG7N+/X+1YQqRI4Vz2lM3jSFyCnvVBd9SOI94iVcWIhYUFnp6e+Pv7J7vf39+f6tWrv/ZzatSowZ07d3j69GnSfRcuXECr1ZInT540RDZt/fopbxcuhJdeMpEZzHTQ+i8o0QLiY2BFR7i8U+1UqnN0dOTvv/+mbt26PHnyhMaNG7Nzp7wuwjhIe3jjkOppmiFDhjBnzhzmzZvH2bNnGTx4MDdu3KBv376AMsXStWvXpMd37NiRHDly0KNHD4KDg9m7dy/Dhg2jZ8+eWFtbp993YiIaNICiReHJE1iyRO00WZCZOfjMhWJNIT4alneAq3vVTqU6Ozs7Nm/ejJeXF5GRkTRr1oytW7eqHUuId2pR1h1zMw1n7kRwLtT0N0MYq1QXI+3bt2fatGmMHTuWcuXKsXfvXrZs2UK+fPkACAkJSdZzxM7ODn9/fx4/fkzFihXp1KkTzZs357fffku/78KEaLXK2hFQpmqkeaAKdBbQdgEUaQxxUbCsPVw7oHYq1dnY2LBhwwaaN29OVFQU3t7erF+/Xu1YQrxVNlsL6hfPBUjPEUOW6j4jasgKfUZe9vgxuLvD8+ewdy/UqqV2oiwqNgpWdFCmasxtoctayFtF7VSqi4mJoXPnzqxatQpLS0t27tz5xmlaIQzB32dC6bP4BM72lhwaWR+dmTRyyiwZ0mdEZA4nJ+jUSbmWhawqMreCj5dBgToQ+wyW+MCt42qnUp2FhQXLli3D29ub6OhovL29uXz5stqxhHijesVykc3GnHtPotl/6b7accRrSDFioBJ7jqxZA29o4SIyg7k1dFgB+WpCzBNY3BruBKqdSnU6nY6lS5dSsWJF7t+/T9OmTXnw4IHasYR4LQudFu9y0nPEkEkxYqDKlYNq1SA2FuQAVZVZ2EBHX8hbDaLDYVFLCDmpdirV2drasnHjRvLmzcuFCxdo1aoV0dHSy0EYJp8Xu2q2nwklIipW5TTi/0kxYsASR0f++AOiotTNkuVZ2kGnVZCnMkQ9hkXecPeM2qlU5+rqypYtW3BwcGDfvn10796d69evy7HtwuCUzu1AkVx2RMclsOVkiNpxxP+RYsSAtWkDuXPD7dswbZraaQSW9tB5NeT2hOcPYWELCDundirVlSpVijVr1qDT6VixYgX58+cnZ86cNGzYkGHDhnHixAm1IwqBRqPBx1N6jhgqKUYMmKUlTJyoXI8fDyFSzKvPyhE6+4FbWYi8Dwubw/2LaqdSXcOGDfH19aVcuXLodDoePnzIjh07mDx5MlWqVOGXX36R0RKhulblc6PVwLFrj7j+4JnaccRLpBgxcB07QpUqSjfWr75SO40AwNoJuqwDlw/gWZhSkDyQ3SStW7cmMDCQp0+fcvz4cWbPnk3z5s2Jj49nyJAhtGvXLkucwC0Ml4uDFTUK5wSk54ihkWLEwGm1/03RzJ8PAQGqxhGJbLJD1/WQqyQ8CVEKkodX1U5lECwtLfH09KR3796sX7+e6dOnY25uzurVq6lcuTJnzshaG6GeNi+mavwCb5GQIKN1hkKKESNQtaoyQqLXw6BB0pXVYNjmgK4bIGcxiLitFCSPrqudyqBoNBo+//xz9u7dS548eTh//jyVK1dm0qRJsvNGqMKrpCt2ljpuPnzOsWsP1Y4jXpBixEhMnAjW1rBvn9J7RBgIO2fotgFyFIbwm0pBEi6L4/5f1apVCQgIoFGjRkRGRjJy5EhKlSrFhg0bZC2JyFTWFmY0/cAVkIWshkSKESPh4QHDhyvXw4bJVl+DYu8K3TZCtgLw+Dos+Agi5Ljy/+fs7My2bdtYtGgRbm5uXL58GW9vbxo3bkxwcLDa8UQWkthzZMupUJ7HxKucRoAUI0Zl+HDIkweuXQM5Z9DAOLhD903glA8eXVVGSJ5I69z/p9Vq6dKlC+fPn2fUqFFYWFjg7+9PmTJlGDBgAI8ePVI7osgCKuXPjkd2a55Gx7E9WP6dGgIpRoyIjQ2MG6dcT54Mz2RnmmFxzKOMkDh6wINLSh+Sp2FqpzJI9vb2/Pjjj5w9e5aWLVsSHx/P9OnTKVKkCLNmzSIuLk7tiMKEabUaWpdXRkdWn5CpGkMgxYiR6dQJChaEe/fgzz/VTiNekS2fUpA45Ib755VOrc/kzJY3KViwIGvXrsXf359SpUrx4MED+vXrh6enJ7t27VI7njBhrSsoZ9UcuHSf0HCZ91abFCNGRqeD0aOV659/hufP1c0jXiN7AaUgsXOFsGClIImUVftv07BhQ4KCgpgxYwbZsmXj5MmT1K9fnzZt2nD1qmyZFukvXw5bKuXPRoIe1gVJzxG1STFihLp0gXz5lNN85RA9A5WjkLKGxDYX3D0Fi1vCc1kP8TY6nY7+/ftz8eJF+vfvj1arZc2aNZQoUYKvvvqKp0+fqh1RmJjEhaxrTtySXV0qk2LECFlYwMiRyvWkSSDtGgxUziLKCIlNTgj5Fxa3hqhwtVMZvBw5cjBjxgyCgoKoX78+0dHRjB8/njp16hATE6N2PGFCmpZxw1Kn5WLYU07dln+bapJixEj16PHfIXrz56udRrxRruJKHxLr7HAnAJa0gegnaqcyCh988AH//PMPfn5+5MiRg4CAACYmHtYkRDpwsDLHq9SLniOykFVVUowYKUvL/0ZHJkwA+YPRgLmUUlrHWznBraOwtC1Ey5RDSmg0Glq1asWMGTMAGDdunLSTF+nK58VC1g3/3iEmLkHlNFmXFCNGrFcvcHWFGzdg0SK104i3cisDXdeBpSPcOATLP4aYSLVTGY327dvTvHlzYmNj6d27N/Hx0qhKpI9aRZzJZW/Jo8hYdp2XrfhqkWLEiFlb/9eVdfx4iJTfbYbNvTx0WQuWDnBtn1KQxMp2qJTQaDTMnDkTBwcHDh8+nDRSIsT7MtNqaFVeGR2RqRr1SDFi5Pr0AXd3pSvr4MFqpxHvlMcTOq8BCzu4ugdWdIJY6XGQEnny5OGnn34CYPTo0bLlV6Sb1i921ew6H8bDZzLnrQYpRoycjY0yRaPRwF9/gZ+f2onEO3lUhk6rwNwGLu+AlV0gTrZEpcQnn3xCnTp1iIyM5NNPP5XtmCJdFHO1p3RuB2Lj9Wz8V86VUoMUIyagQYP/pmt694abN9XNI1IgX3XouBJ01nBxO6zqDnHyF9m7aLVaZs+ejZWVFf/88w9ly5bl119/5cED6XIr3k9SzxE5yVcVUoyYiB9+gEqV4NEj6NwZZH2fEShQCzosBzNLOL8F1vSE+Fi1Uxm8IkWK8Pvvv2NpacmpU6cYNGgQ7u7utG/fnqCgILXjCSPVoqw7Oq2Gk7fCuXhXtt9nNilGTIS5OSxfDnZ2sHcv/Pij2olEihSqBx8vAzMLOLsR/D6BeDkk7l169uxJSEgIM2bMoHz58sTExLBy5Urq1q1LWJjsiBCpl8POkrrFcgGwWkZHMp0UIyakUCGYOVO5/v57OHRI3TwihYo0hHaLQWsOZ9bCus8gQYa23iVbtmz079+fgIAAAgICKFOmDOHh4Xz99ddqRxNGqo2nsqtmXeBt4hNkPVJmkmLExHTpAh07KtM0332ndhqRYsU+hLYLQKuDUythwxeQIA2YUqp8+fJJ231nz54t0zUiTeoVz4WTjTl3I6I5cOm+2nGyFClGTNC4ccrumu3b4exZtdOIFCvxEfjMBY0ZBC2FTQOlIEmFWrVq0b59e/R6PYMGDZKdNiLVLHVmNC/jDshC1swmxYgJKlAAmjdXrqU3lJEp1RJa/wUaLQQsAv+vQX6ppthPP/2ElZUVe/bswU/2uYs08PFUdtX8fSaUJ1GyoDyzSDFiogYMUN4uXAjhchilcfmgDXj/rlwfmgEHpqkax5jkzZuXYcOGATB06FCioqShnEidsnkcKeRsS1RsAltPhaodJ8uQYsRE1a8PJUvCs2dyqq9RKtcRvMYr1/98p4ySiBQZMWIEuXPn5tq1a0ydOlXtOMLIaDSapNER2VWTeaQYMVEazX+jI9OnS98Ro1T9c6gxSLneOBDOblI1jrGwtbVl0qRJAPz444+clYVTIpValc+NRgNHrz7k5kM59CszSDFiwjp3BicnuHIFtm5VO41Ik4bfQfnOoE+A1T3h6j61ExmFjh07Uq1aNZ49e0bZsmUZNmwY4TJfKVLIzdGaGoVyAuAXcFvlNFmDFCMmzNZWaQ8PyuiIMEIaDXz0KxRrBvHRsKw93DiidiqDp9FoWLlyJU2aNCE2NpbJkydTtGhR5s2bR4LsUBIp0LqC0nNkfdBt2ZmVCaQYMXH9+4NWK9t8jZqZDtrMhQJ1IPYZLPGBW8fVTmXw8uTJw5YtW9i8eTNFixYlLCyMXr160bBhQ1nYKt7Jq5QrljotV+4/48ydCLXjmDwpRkxc/vzQooVyLdt8jZi5NXRYAflqQswTWNwa7gSqncooNG3alFOnTjFlyhTs7OzYtWsX3bp1kxES8VZ2ljoalnABkJN8M4EUI1nAF18obxcuhFDZqWa8LGygoy/krQbR4bCoJYScVDuVUbCwsGDIkCFs2LABc3NzVq5cyZgxY9SOJQxc87JuAGw6GUKCtIfPUFKMZAH16ikn+j57BnJsh5GztINOqyBPZYh6DIu84e4ZtVMZjXr16jF79mwAJk6cmHQtxOvULZYLO0sdtx8/J+DGI7XjmDQpRrIAjQZ++UW5njsXAmV037hZ2kPn1ZDbE54/hIUtIOyc2qmMRrdu3fjmm28A+Oyzz9i+fbvKiYShsjI3w6uUTNVkBilGsogaNaB9e6Wz+ODB0mHc6Fk5Qmc/cCsLkfdhYXO4f1HtVEbju+++o3PnzsTHx9OmTRsCAgLUjiQMVPOyylk1m0+FEBcv64wyihQjWchPP4GVFezZA2vXqp1GvDdrJ+iyDlw+gGdhSkHy4LLaqYyCRqNhzpw51KlThydPntCgQQOOHJEt0+JVNQvnJJuNOfefxnD4ykO145gsKUaykLx54cWxHQwdCrK70QTYZIeu6yFXSXgSohQkD6+qncooWFpasmHDBmrUqMHjx49p1KgR+/fvVzuWMDDmZlqafKAsZJWpmowjxUgWM3w4uLvD1avw669qpxHpwjYHdN0AOYtBxG2lIHl8Q+1URsHBwYFt27ZRr149njx5QuPGjdmxY4fasYSBaV5GmarZejqE6Dg5WyMjSDGSxdjZwcSJyvW4cbLV12TYOUO3jZCjMITfhAUfQbgc8pUSdnZ2bN68mQ8//JDIyEiaNWvGVjk/QbykcoHsuDhYEhEVx74L99WOY5KkGMmCOnWCypXh6VPo21cWs5oMexelIMlWAB5fV0ZIImRYOSWsra1Zt24d3t7eREdH07ZtW65du6Z2LGEgzLQamn2gjI5sPCn/pjKCFCNZkFYLM2eChQWsXw9yyroJcXCH7pvAKR88vKIUJE/uqp3KKFhaWrJq1Spq1arFs2fP6NOnj5xJIpK0KKcUI/7Bd3keI1M16U2KkSzK0/O/3iMjRsCBA+rmEenIMY9SkDjmhQeXlILk6T21UxkFc3Nz5syZg5WVFdu3b2fRokVqRxIGomweR/JmtyEyJp7twTK/nd6kGMnCPvsMPv4Y4uOVHiT35PeV6XDKC902gENuuH8eFrWAZw/UTmUUihYtynfffQfA4MGDuXtXRpaEsh28ZXnlJN81AbdVTmN6pBjJwjQa+OsvKFYMbt9W1pLEy+ij6cheQFlDYu8GYcFK6/hI6ZOQEl9++SUVKlTg0aNHfJF4uJPI8nwqKMXI/ov3uBshvRHSkxQjWZy9PaxZAzY24O8PP/ygdiKRrnIUUgoS21xw9xQsbgnP5YyNd9HpdMydOxczMzNWrVrFWukSKIB8OWyplD8bCXpYGyijI+lJihFBqVLwxx/K9dix4Oenbh6RznIWUQoSm5wQ8i8sbg1R4WqnMnjlypVjxIgRAPTr149Hj6SIE9C6Qh4A1py4JQuc05EUIwKALl3g88+Vbb6dO8PRo2onEukqV3FlDYl1drgTAEt8IPqJ2qkM3tdff02xYsUIDQ1lqmw7E0CzMm5Y6rRcDHvK6dsRascxGVKMiCS//AJNm8Lz59C8OUibBRPjUkppHW/lBLeOwdK2EP1U7VQGzcrKih9ezF3OnTuXuLg4lRMJtTlYmeNVyhWANQHSWDC9SDEikuh0sGIFlC0LYWHQrBmEy2i+aXErA13XgaUj3DgEyz+GmEi1Uxk0b29vnJ2dCQkJYfPmzWrHEQYgcSHr+qDbxMTJSb7pQYoRkYy9PWzapJxfExwMbdtCbKzaqUS6ci8PXfzAwh6u7VMKktjnaqcyWBYWFvTo0QOAv/76S+U0whDULJwTZ3tLHkXGsut8mNpxTEKaipGZM2dSoEABrKys8PT0ZN++fSn6vAMHDqDT6ShXrlxanlZkkjx5YOPG/3bYDBmidiKR7vJUhM5rwMIOru6BFR0hVrYqvknv3r0B2Lp1KzduyCGEWZ3OTEurFz1H/GSqJl2kuhjx9fVl0KBBjBkzhsDAQGrVqkWTJk3e+Q80PDycrl270qBBgzSHFZmnQgVYtky5njEDfH3VzSMyQN4q0GkVmNvA5Z2wsivERaudyiAVKVKE+vXro9frmTNnjtpxhAHwebGrZue5MB49i1E5jfFLdTEydepUevXqRe/evSlRogTTpk3Dw8ODWbNmvfXz+vTpQ8eOHalWrVqaw4rM5e0NI0cq1717w/nz6uYRGSBfdejoCzpruPg3rOoB8TIv9zqffvopIAtZhaKYqz2lczsQG69nw79yeN77SlUxEhMTw4kTJ/Dy8kp2v5eXFwcPHnzj582fP5/Lly/z7bffpuh5oqOjiYiISHYT6vjhB6hTRznht00biJS1jqanQG3osBzMLOH8ZljdUwqS12jZsiU5c+bkzp07bNmyRe04wgAkjo7Irpr3l6pi5P79+8THx+Pi4pLsfhcXF0JDX39w0MWLFxk5ciRLly5Fp9Ol6HkmTJiAo6Nj0s3DwyM1MUU60ulg+XJwcYHTp6FfP6UXiTAxherBx8vAzALObgC/TyFe/vp/maWlJd27dwdkIatQtCjrjk6r4eStcC7elb497yNNC1g1Gk2y9/V6/Sv3AcTHx9OxY0e+//57ihYtmuKvP2rUKMLDw5NuN2/eTEtMkU7c3JQtv1otLFwI8+apnUhkiCINod1i0JrDGT9Y9xkkyGFFL/vkk08AWcgqFDnsLKlbLBcgh+e9r1QVIzlz5sTMzOyVUZCwsLBXRksAnjx5wvHjx/n888/R6XTodDrGjh3Lv//+i06nY+fOna99HktLSxwcHJLdhLrq1oVx45Tr/v3h+HFV44iMUuxDaLcQtDo4tRI2fAEJ0kchUdGiRalbty4JCQnMk6pcAG08lV01awNvEZ8gw8ZplapixMLCAk9PT/z9/ZPd7+/vT/Xq1V95vIODA6dOnSIoKCjp1rdvX4oVK0ZQUBBVqlR5v/QiU40YoXRmjY6GVq1ATlY3UcWbQZt5oDGDoKWwaaAUJC9JXMg6c+ZMLl26pHIaobZ6xXPhaG3O3YhoDly6r3Yco5XqaZohQ4YwZ84c5s2bx9mzZxk8eDA3btygb9++gDLF0rVrV+WLa7WULl062S1XrlxYWVlRunRpbG1t0/e7ERlKq4XFi6FYMbh1S1nQGiM72kxTSW/wmQ0aLQQsgi1fymKhF1q3bk3JkiW5d+8etWvX5uzZs2pHEiqy1JnRoqw7ID1H3keqi5H27dszbdo0xo4dS7ly5di7dy9btmwhX758AISEhMhcqglzdIT168HBAfbvh4ED1U4kMkxpH2j5B6CB4/Ng63ApSFCmkXfu3Enp0qUJCQmhTp06nDx5Uu1YQkU+nsqumm1nQnkSJTvR0kKjN4IzkCMiInB0dCQ8PFzWjxiIzZuVKRu9Hv78E16MXAtTFLgU1vcH9FC1PzQeD69ZsJ7V3L9/Hy8vLwIDA8mePTvbt2/H09NT7VhCBXq9noZT93D53jN+8ilDu0qyAzRRSn9/y9k0Ik2aNftvQevnn0MKTwQQxqh8J2j+q3J9+Hf451sZIUFZ0L9z506qVKnCw4cPadCgwRsX5QvTptFoaC09R96LFCMizUaNUtaNxMZCixZKHxJhojy7QbOpyvWBX2HnD1KQAE5OTvj7+1OrVi3Cw8Np1KgREydOJEEW/GY5rSvkRqOBI1cfcvOhdIdMLSlGRJppNErfkerV4fFjaNwYrl1TO5XIMJV6QZOflOt9U2DPJHXzGAh7e3v+/vtvevToQUJCAqNGjaJly5Y8evRI7WgiE7k5WlOjUE4A/KTnSKpJMSLei42NcsJvqVJw5w54eUGYnKhtuqr0Aa/xyvXuCbB3srp5DIS1tTXz5s1jzpw5WFpasnHjRjw9PQkMDFQ7mshErSu8OMk38BZGsBzToEgxIt5b9uzw99+QNy9cvAhNm8IT6Yxsuqp/Dg2/U653/qBM2wgAevXqxcGDB8mfPz9Xr16lXr163L4tfyVnFR+WdsXWwozrDyI5fl1GxlJDihGRLnLnhu3bIWdOOHECWraUQ/VMWs3BUO8r5dr/Gzj0u7p5DEiFChUICAjA09OT8PBw+vXrJ38lZxE2FjqafOAGSM+R1JJiRKSbYsVg61aws4OdO+HDD0EOXDZhdYZB7eHK9d+j4YgcHpcoW7ZsLFiwAHNzczZs2MCqVavUjiQySeJJvpv+DSEqVs52SikpRkS6qlhRmbJxdFS2+zZoAA8eqJ1KZJh6o5VREoCtw5TmaAKA0qVLM3r0aAC++OILHsg/hCyhSoHs5Hay5kl0HP7BcmZGSkkxItJd9erKyEjOnMqBenXqQEiI2qlEhtBooMG3UP0L5f1Ng5X28QJQjscoWbIkYWFhDBkyRO04IhNotZqkhazScyTlpBgRGaJCBdi7F9zd4cwZqFULrl9XO5XIEBoNNPoBqnymvL9hAAQtUzeTgbC0tGTu3LloNBoWLVrEtm3b1I4kMkFiA7S9F+4RFhGlchrjIMWIyDAlSihTNQUKwOXL0LChbPs1WRoNfDgBKn0C6GFdPzgp6yQAqlatysAXhzj16dOHJ7LVzOQVyGlLhbxOJOhhfdAdteMYBSlGRIYqWFApSPLnh0uXoEkTWdRqsjQapSmaZw9AD2s/hdN+aqcyCOPGjSN//vzcuHEjaR2JMG2Jh+etCZCeIykhxYjIcInbfp2dISBA2fYbJSOXpkmrVdrGl+8M+gRY0xuCN6idSnW2trb89Zey2+j333/nwIEDKicSGe2jD9yx0Gk5F/qEM3fkL7B3kWJEZIoiRWDbNrC3h127oFMniJddb6ZJq4Xmv0GZj0EfD6t7wLktaqdSXaNGjejRowd6vZ7evXsTJRW5SXO0MadRCRdA2sOnhBQjItNUqADr1oGFBfj5Qb9+ctaaydKaQcuZULoNJMTByq5wYbvaqVQ3ZcoUXF1dOXfuHOPHj1c7jshgPp7Krpr1QbeJjZfDE99GihGRqerXh2XLlOUFf/0F86QthenSmkGrP6FkS0iIBd/OcGmH2qlUlS1bNn7/XelWO3HiRP7991+VE4mMVLuIMzntLHjwLIY95++pHcegSTEiMp2PD/z4o3I9YACcO6duHpGBzHTgMweKfwTx0bCiI1zZrXYqVbVu3RofHx/i4uLo1asXcXFxakcSGURnpsW7nPQcSQkpRoQqhg9XRkkiI6FDB4iOVjuRyDBm5tBmPhRtAnFRsOxjuLZf7VSqmjFjBk5OTpw4cYJffvlF7TgiAyW2h99xNozHkTEqpzFcUowIVWi1sHgx5MgBQUEgux1NnM4C2i2Ewo0g7jksbQfXD6mdSjWurq5JRchXX33F/v1ZuzgzZSXdHSjh5kBMfAIbT0or6jeRYkSoxt0d5s9XrqdOVXbbCBOms4T2S6BgPYh9BkvbwM1jaqdSTbdu3WjdujUxMTF4e3tz8eJFtSOJDOKT2B7+hEzVvIkUI0JVzZvD558r1926wV05V8q0mVvBx8sgfy2IeQpLWsPtE2qnUoVGo2Hx4sVUrlyZhw8f0rRpU+7fv692LJEBvMvlxkyrIejmYy7fe6p2HIMkxYhQ3c8/wwcfKK3iK1WSERKTZ2EDHX0hXw2IjoDFreBOkNqpVGFjY8OGDRvInz8/ly5domXLltJ/xAQ521tSp6gzAH6ykPW1pBgRqrOyglWrlNbxN28qLeN79IBHj9ROJjKMha1SkHhUgahwpSC5d0HtVKpwcXFh8+bNODo6cuDAAbp3705CgvSkMDWJC1nXBtwmIUEaLP0/KUaEQShWDE6ehEGDlB4kCxZAyZJKkzRhoiztodNqcC8Pzx8qBUl41uxUWbJkSfz8/NDpdPj6+jJgwAApSExMgxK5cLDScSc8ikNXHqgdx+BIMSIMhq0t/PIL7N+vFCehodCqFfTpo2wBFibIykEpSHIUgYhbSkES+VDtVKqoX78+8+bNQ6PR8Pvvv9OnTx/i5cwEk2FlbsZHZd0BWcj6OlKMCINTvbqy3XfYsP86tVaqBKdOqZ1MZAjbnNDFD+zd4f55WNYOYp6pnUoVXbp0YeHChWi1WubMmUP37t2lKZoJSZyq2Xo6lKfR8nN9mRQjwiBZWcFPPymn/bq6QnAwVK4Ms2bJeTYmySmvUpBYOcGtY8pZNnFZs0FUly5dWL58OTqdjiVLltCxY0diY2PVjiXSQYW8ThTIacvz2Hi2nQ5VO45BkWJEGLSGDZW1JE2bQlSUcrhely7SsdUk5SoBnVaBuQ1c+gfW9IL4rPlLuF27dqxevRoLCwtWrVpF+/bt0UsVbvQ0Go30HHkDKUaEwXN2hk2blPUkOh0sXQqNG8tuG5PkURnaLwYzCzi7Afw+hfisOZzt7e3N+vXrsbS0ZO3ataxcuVLtSCIdtHoxVXPoygNuPZLFcImkGBFGQaNRdtps3Qr29rBnD9SoAdeuqZ1MpLvCDaHdYtCawxk/WPcZJGTNhZwffvghY8aMAWD48OFEykpuo5fbyZpqBXMAsC4wa+4eex0pRoRRadhQ2W2TOzecPQvVqsGJrNnA07QV+xDaLgCtDk6thA1fQBbd6jp06FDy5s3LjRs3mDx5stpxRDrw8VRGR9YE3JbptxekGBFGp0wZOHxYeRsaCrVrw7x5srDV5JT4CHzmgsYMgpbCpoFZsiCxtrbmp59+AmDSpEncuiVrDYxdk9KuWJubcfX+MwJuPFY7jkGQYkQYpTx5YN8++PBDpQdJr17QoQOEh6udTKSrUi2h9V+g0ULAItgyNEtWne3ataNmzZpERkYycuRIteOI92RrqaNJaVcA1kh7eECKEWHEHBxg82aYOFFZ2OrrC+XKKaMmwoR80AZa/gFo4Phc2DoiyxUkGo2GX3/9FY1Gw9KlSzl06JDakcR7Spyq2fTvHaJis+aaqJdJMSKMmlYLI0Yo60gKFFAWtNasCZMnZ7nfV6atbHvwnqFcH/0Ttn+V5X7AFSpUoEePHgAMHDhQ2sUbuWoFc+DuaEVEVBw7zoapHUd1UowIk1ClCgQGwscfQ3y80r21UydpI29SyneG5r8q14dmwD/fZbmCZPz48djb23Ps2DGWLl2qdhzxHrRaDS3Lv+g5IlM1UowI0+HoCMuWwe+/K9M2y5crreWvXlU7mUg3nt2h6YsdJQemwa7xaqbJdK6urowePRpQChM5u8a4JU7V7Llwj3tPsnYnRylGhEnRaJQurTt2QK5c8O+/ULEi/POP2slEuqn8CXw4Sbne+zPsnqRunkzWv39/nJycOH/+POvXr1c7jngPhZztKOfhRHyCnvVBWbvniBQjwiTVrg3HjyuFyMOHyq6b1avVTiXSTdW+4DVOud79I+zNOv037O3t+fzzzwGYMGGC9Kkwci/3HMnKpBgRJsvDQ9n+26GDso6kQweQPyRNSPUvoMG3yvXOH+DAr+rmyUQDBgzA2tqa48ePs3PnTrXjiPfQvIwbFmZazoZEEHwnQu04qpFiRJg0KytYvBg6doS4OGjbVjnnRpiIWkOgntIuHf9v4NDv6ubJJM7OzvTu3RtQRkeE8XKysaBBiVwA+GXhhaxSjAiTZ2YGCxdCu3YQGws+PrBtm9qpRLqpMxxqD1eu/x4NR/5SN08m+fLLL9HpdOzYsYNjx46pHUe8B58Xh+etC7pDXHzW3LItxYjIEnQ6WLIEWreGmBho2RL+/lvtVCLd1BsNNQcr11uHwfF56ubJBPny5aNjx44ATJw4UeU04n3UKeZMDlsL7j+NZt/F+2rHUYUUIyLLMDdXtvs2bw7R0fDRR7BggdqpRLrQaJT1I9WUhZ1sGqy0jzdxI0aMAGDt2rWcO3dO5TQirczNtLQo5w7A6iw6VSPFiMhSLCxg1SplMWtcHPToAWPHZrneWaZJo1F22FT5THl/wwAIWqZupgxWsmRJWrZsiV6vTzpMTxinxKka/+C7hEfGqpwm80kxIrIcS0tlymbUKOX9b7+Fnj2V9STCyGk08OEEqNQb0MO6fnByldqpMlTiwXmLFi1i+/btKqcRaVXK3YFiLvbExCWw6dQdteNkOilGRJak1cKPP8IffyjXCxZAkyZw/braycR702igyc9Kt1b0sPZTOO2ndqoMU6VKFbp160Z8fDxt2rTh1KlTakcSaaDRaPDxVNrD+2XBniNSjIgsrU8f2LgRbG2Vrq3FiysjJXKmjZHTaqHZL1CuM+gTYE1vCN6gdqoM8+eff1K7dm2ePHlCs2bNuHMn6/1lbQpalsuNVgMnrj/i6v1nasfJVBq9EbTvi4iIwNHRkfDwcBwcHN74uPj4eGJlrN0omZubY2ZmptrzBwfD55/Drl3K+x4e8PPPynZgjUa1WOJ9JcS/mKpZAVodtFsMxZuqnSpDPHz4kOrVq3P+/HkqVKjAnj17sLOzUzuWSKVu846y58I9vqhfmC+9iqkd572l9Pe3SRQjer2e0NBQHj9+nPnhRLpxcnLC1dUVjUq//fV6WLMGhg79b7qmXDkYNEg5DdjSUpVY4n0lxIPfJ3B6DWjN4eNlUNRL7VQZ4vLly1StWpX79+/TvHlz1q5dq2qRL1Jvw793GLA8kNxO1uwbXg+t1rj/GspSxUhISAiPHz8mV65c2NjYqPbLTKSNXq8nMjKSsLAwnJyccHNzUzXP8+fKqMikSf9N1+TKBZ99Bn37gqurqvFEWsTHwZqeELwezCyhw3Io3EDtVBni0KFD1KtXj+joaCZNmsTw4cPVjiRSISo2nkrj/uFJdBzLP6lKtUI51I70XrJMMRIfH8+FCxfIlSsXOXIY9w8tq3vw4AFhYWEULVrUIP6ae/gQZs+GGTPg1out/1otVKsGTZtCs2ZQpoxM4xiN+FhY1R3ObQKdFXT0hYJ11U6VIf766y/69OlD3rx5uXLlikH8exIpN3LNSVYcu0kbzzxMbltW7TjvJaXFiNEvYE1cI2JjY6NyEvG+En+GhrLuJ3t2GDECrlyBFSugalVISIADB2DMGGUKx8MDBg5UTgg2/LI+izMzhzbzoWgTiIuCZR/Dtf1qp8oQXbt2JVu2bNy4cUO2+xqhxJN8t54KITImTuU0mcPoi5FEMjVj/Az1Z2huDu3bw6FDylqSWbOULq42NnD7Nvz2G1SqBKVKwYQJcPOm2onFG+ksoN1CKNwI4p7D0nZw/ZDaqdKdlZUV3bp1A5RREmFcKubLRr4cNjyLiWfb6VC142QKkylGhMgMefMq60Y2bIAHD5RtwR9/rJwOfPYsjB4NBQvCd98pZ+AIA6SzhPZLoGA9iH0GS9vATdM7aO6TTz4BYOPGjbLV18hoNBpal1dGR7JKzxEpRoRIIysr5Xyb5cshNBTmzoVatZQ2899/DxUrKtM3wgCZWym7agrUhpinsKQ13D6hdqp0VbJkSWrWrEl8fDzz589XO45IpdYVlAZoBy7f587j5yqnyXhpKkZmzpxJgQIFsLKywtPTk3379r3xsX5+fjRq1AhnZ2ccHByoVq0af2fx41I1Gs1bb927d3/lcfb29lSsWBE/v/86SX733XdJH9dqtbi7u9OpUyduyjxBpnN0VFrK79kDvr6QMyecOgVVqsDIkdJEzSBZ2ECHFZCvBkRHwOJWcCdI7VTp6tNPPwVg9uzZJCRkzaPpjZVHdhsqF8iOXg9rA01/dCTVxYivry+DBg1izJgxBAYGUqtWLZo0acKNGzde+/i9e/fSqFEjtmzZwokTJ6hXrx7NmzcnMDDwvcMbq5CQkKTbtGnTcHBwSHbfr7/+mvTY+fPnExISwrFjxyhbtixt27bl0KH/5rhLlSpFSEgIt27dwtfXl1OnTtGuXTs1vi2BsrOmXTuliVqHDsqC10mTwM0NeveGvXuV+4SBsLBVdtV4VIGocFjcEkJNp516mzZtcHJy4vr167KQ1Qi1eXF43pqAWxjBxtf3o0+lypUr6/v27ZvsvuLFi+tHjhyZ4q9RsmRJ/ffff5/ix4eHh+sBfXh4+Csfe/78uT44OFj//PnzFH89QzJ//ny9o6Pjaz8G6NeuXZv0fkxMjN7Gxibptf7222/1ZcuWTfY5v/322xtfK0Nn7D/L11m3Tq8vUECvV/baKLd8+fT6r7/W62/dUjudSPI8XK//q75e/62DXj+pgF4fekbtROlmwIABekDfqlUrtaOIVIp4HqMv9tUWfb4Rm/SBNx6pHSdN3vb7+2WpGhmJiYnhxIkTeHkl717o5eXFwYMHU/Q1EhISePLkCdmzZ3/jY6Kjo4mIiEh2Sym9Xs+zZ89UuekzuHI1NzdHp9O9cetraGgofn5+mJmZSV8BA+HtDZcuwe7d0KsXODgoO3J++AHy54dOneDoUbVTCqwcoPMacC8PkQ9gUQu4d17tVOkicSHrhg0bCAkJUTmNSA17K3M+LKV0WVxz4pbKaTJWqoqR+/fvEx8fj4uLS7L7XVxcCA1N2fajKVOm8OzZs7dOJUyYMAFHR8ekm4eHR4ozRkZGYmdnp8otMgMXBkRHRzNu3DgiIiJo0OC/zpGnTp3Czs4OGxsb3Nzc2L17N/3798fW1jbDsojU0WqhTh2YM0dZ6LpihfJ+XBwsW6asK6leHf7667/makIF1k7Q2Q9cy8Cze7CwOdy/pHaq91a6dGmqV68uC1mNVOsXUzUb/r1DdFy8ymkyTpoWsP5/Pwi9Xp+iHhHLly/nu+++w9fXl1y5cr3xcaNGjSI8PDzplpUXZHbo0CGp2Jg6dSqTJ0+mSZMmSR8vVqwYQUFBHDt2jPHjx1OuXDnGjx+vYmLxNtbWSs+S3bshIAC6dQMLC6WHSZ8+ShO1smVh1Cg4Znq7TQ2fTXbouh5ylYKnd5WC5OEVtVO9N1nIarxqFM6Jq4MV4c9j2Xk2TO04GSZVxUjOnDkxMzN7ZRQkLCzsldGS/+fr60uvXr1YuXIlDRs2fOtjLS0tcXBwSHZLKRsbG54+farKLSO6wP7yyy8EBQUREhLCw4cP+fLLL5N93MLCgsKFC1OqVClGjx5NuXLl+Oyzz9I9h0h/5cvDggVw4wb8+KPSZl6jgZMnYeJEqFxZ6Vdi6uvWDE5iQeJcHJ7cgQXN4dE1tVO9l7Zt2+Lk5MS1a9fYsmWL2nFEKphpNbQsr2zzXWPCPUdSVYxYWFjg6emJv79/svv9/f2pXr36Gz9v+fLldO/enWXLltGsWbO0JU0hjUaDra2tKreM6CDq6upK4cKF3zqS9LKvv/6a5cuXExAQkO5ZRMZwcVFGQg4ehLAwWLIEfHyUj33/PfToIQ3UMp2dM3TdADmKQMQtZYTksfGO0NrY2NCrVy8AfvvtN5XTiNRq46kUI7vPh/HgabTKaTJGqqdphgwZwpw5c5g3bx5nz55l8ODB3Lhxg759+wLKFEvXrl2THr98+XK6du3KlClTqFq1KqGhoYSGhhIeHp5+34VIUrBgQby9vfnmm2/UjiLSIGdOZVHr6tXwxx/KepOFC5VD+eSfTCazd4FuGyF7IXh8AxZ+BOHG+5fp559/jlarxd/fn+DgYLXjiFQonMuesnkciUvQsz7INLvpproYad++PdOmTWPs2LGUK1eOvXv3smXLFvLlywcoPTRe7jny559/EhcXR//+/XFzc0u6DRw4MP2+C5HMl19+yebNmzly5IjaUcR76NNHaTdvawv//KN0d71+Xe1UWYyDm1KQZMuvTNUsbA4RxrkjJX/+/Hh7ewMwffp0ldOI1Gr9Us8RU6TRZ/R+1HTwtiOIo6KiuHr1alJHWGG85Gf5egEByshIaKjSgr5vX6Wr6zuWaYn09PgmLGiqjJDkLArdNikjJ0Zm9+7d1KtXDxsbG27dukW2bNnUjiRS6NGzGCr/+A+x8Xq2DapFcdeUr6VU09t+f79MzqYRwsBVqACHD0PNmhAVBdOmKYfxjRgB9++rnS6LcPJQChCHPHD/gtKH5Ok9tVOlWp06dShTpgyRkZHMnTtX7TgiFbLZWlC/uLJ20BQPz5NiRAgjkC+f0kr+77+VviSRkfDTT0pRsnmz2umyiGz5oPtGsHeHe+dgkTc8e6B2qlTRaDQMGDAAgBkzZhAXF6dyIpEaPi+matYG3iYu3rS2aEsxIoSR0GjAy0vpSbJ5s7I1+MkTaNEC/vxT7XRZRPaCyhoSO1cIOwOLvSHyodqpUqVjx47kyJGD69evs3HjRrXjiFSoWywX2WzMufckmv2XTGtYVIoRIYyMRgNNm8KRI8q234QEZR3JqFFyCF+myFlYKUhsnZVD9Ra3gueP1U6VYtbW1klN0GSbr3Gx0GnxLmeaPUekGBHCSJmbw9y5Si8SUBqlde4M0abZhsCwOBdVChKbHBASBEt8ICrlZ2iprV+/fpiZmbF7925OnjypdhyRColTNdvPhBIR9fpzyoyRFCNCGDGNBr75RunkqtPB8uXKgtfVq2WUJMPlKqF0arXOBrePw9I2EP1E7VQpkidPHnxedNb77LPPiIqKUjmRSKnSuR0oksuO6LgEtpw0zm3mryPFiBAmoFs32LoVsmWD4GBo21YpSjZskHbyGcr1A+iyDqwc4eYRWNoOYp6pnSpFfvjhBxwdHTl48CA9evSQM2uMhEajwcfT9HqOSDEihIlo2BCuXFFGSuzt4d9/wdsbypRR1pZMmKCMmJw8qZwYLNKJeznoshYsHeDGQVjWHmIy7gTv9FK0aFH8/PzQ6XSsWLGCr7/+Wu1IIoValc+NVgPHrj3i+gPjKH7fRYoRIUyIk5OyhuTqVWVBq60tnD6tTOOMHq2MmJQtC0WLwvz5UpSkm9ye0NkPLOzh2j5Y0RFiDX/qo379+syZMweAH3/8MelaGDYXBytqFM4JmE7PESlGhDBBOXIoJwFfuwYrVsAPP0CXLlC1Kjg4KMVKz55QooRyMF98vNqJTYBHJei8Gsxt4cou8O0McYa/mrhbt25JoyJ9+/Z95SBUYZjavJiq8Qu8RUKC8c/FSjGiou7du6PRaF65Xbp0KdnHzM3NKViwIEOHDuXZM2VI7tq1a8k+x9HRkapVq0rfAJFMzpzQvj189RUsWqT0KAkJgZ9/Vj526ZJSpBQrBp99phzKd/68LH5Ns7xVodNK0FnDJX9Y1R3iDX/Hw/fff0+nTp2Ij4+nbdu23LtnfN1lsxqvkq7YWeq4+fA5x64ZV6+b15FiRGUffvghISEhyW4FChRI9rErV64wbtw4Zs6cydChQ5N9/j///ENISAhHjhyhcuXK+Pj4cPr0aTW+FWEkbGxg6FBlfcmPPyqLXi9fVk4J7t4dihdXCpWvvpKiJE3y14SOvqCzgvNbYMMXBv9CajQa5s6dS/ny5QkPD2fixIlqRxLvYG1hRtMPXAHTWMgqxYjKLC0tcXV1TXYzMzNL9jEPDw86duxIp06dWLduXbLPz5EjB66urhQvXpzx48cTGxvLrl27VPhOhLGxt1fWlVy7BqtWKQVKzZrKYXyPHsH48dC7t0zhpEnBOtB2IWjM4N/l4P+1wW9rsrS0ZMKECQD8/vvv3L5tGmsRTFliz5Etp0J5HmPc/1BNrhjR6+HZM3VuGf3/Gmtra2JjXz/kGxsby+zZswEwNzfP2CDCpDg4QJs2ytTNvn0QEQFz5oBWqyxy7dABYmLUTmmEin0I3r8r14dmwIFf1c2TAl5eXtSsWZPo6GjGjx+vdhzxDpXyZ8cjuzVPo+PYHhyqdpz3YnLFSGQk2Nmpc4tMw26+TZs2YWdnl3Rr27btax939OhRli1bRoMGDZLdX716dezs7LCysuLLL78kf/78tGvXLi0vnRCA0tm1Vy9ltMTcXHnbqhU8f652MiNUrgN4jVOu//kWAharm+cdNBpNUhEye/Zsrl69qnIi8TZarYZW5ZXRkdUnjHuqxuSKEWNTr149goKCkm4vnxWRWKhYWVlRrVo1ateuzfTp05N9vq+vL4GBgWzYsIHChQszZ84csmfPntnfhjBBrVvDxo1gbQ1btkCTJiDrGtOg+hdQY5ByvXEABG9QNc671K5dm0aNGhEXF8f3iWcNCIPlU0E5q+bApfuEhhv+dvI30akdIL3Z2MDTp+o9d2rZ2tpSuHDh136sXr16zJo1C3Nzc9zd3V87/eLh4UGRIkUoUqQIdnZ2+Pj4EBwcTK5cuVIfRoj/07gx/P03NGsGe/ZAyZIwYwa0a6e0ohcp1PA7iLwPgUtgdQ9otxiKN1U71RuNGzcOf39/Fi9ezMiRIylevLjakcQb5MthS6X82Th27RHrgm7Tt04htSOlicmNjGg0SqMnNW7p/T/nxEIlX758KVoHUqdOHUqXLi1zvSJd1aqlrCX54AO4fx8+/lgZNQkxnWMxMp5GA81/g9JtICEOVnaFC9vVTvVGlStXxtvbm4SEBL799lu144h3SFzIuubELfQGvlD6TUyuGMnqvvzyS/78809ZCS/SVdmycPw4fPedciDfunXKKMmUKXD3rtrpjITWDFr9CSVbQkKs0hTt0g61U73R2LFj0Wg0rFy5kqCgILXjiLdoWsYNS52Wi2FPOX3beE6PfpkUIybmo48+In/+/DI6ItKdhQV8+y2cOAGenvD4sbIdOHduaNpU6fQqi1zfwUwHPnOg+EcQH620jb+yW+1Ur1WmTBnat28PQJcuXQgPD1c5kXgTBytzvEoZd88Rjd4IxnQiIiJwdHQkPDwcBweHZB+Liori6tWrFChQACsrK5USivQgP0vjERcHc+fCvHlw9Oh/9zs4KG3mv/gCChZUL5/Bi4t5MVWzVenW2nm10izNwNy6dYvKlSsTEhJCo0aN2Lx5s7QOMFC7z4fRff4xstmYc2R0Qyx0hjHW8Lbf3y8zjLRCCKOi00GfPnDkCJw7p3RrzZ9f6VEybRoULqxsB96zx+B7falDZwHtFkLhRhD3HJa2g+uH1E71ijx58rBx40ZsbGzw9/enX79+RrsmwdTVLJwTZ3tLHkXGsvt8mNpxUk2KESHEeylWTDmI7/Jl2LYNPvxQKUDWrYO6daFIERg4ELZvhyjj3XmY/nSW0H4JFKwHsc9gaRu4eUztVK/w9PRkxYoVaLVa5syZw08//aR2JPEaOjMtrcor23yNcapGihEhRLrQapWtwFu3QnAw9O2r9Ci5fBl++035WI4cyk6cM2fUTmsgzK3g42WQvxbEPIUlreH2CbVTvaJ58+ZMmzYNgJEjR7Jy5Up1A4nXStxVs/NcGI+eGVfbZClGhBDprkQJmDVL2Wmzdq1yxo27u9KleO1aqFBBOfvmDacbZC0WNsrBevlqQHQELG4Fd4LUTvWKL774goEDBwLQrVs3zp8/r3Ii8f+KudpTyt2B2Hg9G/69o3acVJFiRAiRYeztoWVLmD0bbt2CgAD46CPlrJuvvoLKlUF2jQIWtkpB4lEFosJhcUsIPaV2qldMmTKFRo0aERUVRe/evUkw8NOIs6KkniNGNlUjxYgQIlNoNFC+PGzYAEuWQPbsSiFSqRK0bQtffw2LF8Phw8qpwVmOpT10Wg25K8LzR7DIG+4Gq50qGTMzM2bPno2trS379+/njz/+UDuS+D/e5dzRaTWcvBXOxbtP1I6TYlKMCCEylUYDnTop60ratFG2Ca9eDePGQdeuUK0a5MwJPj5K59cstXnDygE6rwH38hD5ABa1gHuGNR2SL18+Jk6cCMCIESO4ceOGyonEy3LYWVK3mHIcyJoA42l+KcWIEEIVLi7KicB79sDPP8Onn0K9ekoTtYQE8POD2rWhYkVYtChtp2IbJWsn6OwHrh/As3uwsDncv6R2qmT69etHjRo1ePr0KX379pXtvgYm8fC8tYG3iE8wjp+NFCNCCFXVrq10cv3zT9i5U1lbcvq0UpxYWSnrTLp1U85/8vCABg2UnTq//grXr6udPoPYZIeuG8ClNDy9Cws/ggeX1U6VJHGbr4WFBVu3bmXZsmVqRxIvqV8iF47W5tyNiObApftqx0kRKUayoO+++45y5cq99THdu3enZcuW6fq8GfE1hWkqVUopTm7dgh9/VIoQUN7fuVP52KBBSqO1evWUTrARxnkkx5vZZIeu68G5BDwJgYUt4NE1tVMlKV68ON988w0AAwcOJCzM+BptmSpLnRktyroDxrOQVYoRFWg0mrfeunfv/srj7O3tqVixIn5+fklf57vvvkv6uFarxd3dnU6dOnHz5s23Pv/QoUPZscNwD+gSIlGOHDBqlDICcu8eHDwICxbAmDFKQzWA3buhVy9l2qdnT7hyRcXA6c02J3TbADmLQsQtWNAcHhvOGo3hw4dTpkwZHjx4QKVKlVi5cqVM2RgIH09lV83fZ0J5EmX4e+ilGFFBSEhI0m3atGk4ODgku+/XX39Neuz8+fMJCQnh2LFjlC1blrZt23Lo0H9to0uVKkVISAi3bt3C19eXU6dO0a5du7c+v52dHTly5Miw70+I9KbRKItaq1VTpmzGjYNdu5Qi5ccflb4mUVEwf77SEfaTT0xoCscuF3TbCNkLQfgNZQ1JuGEsTDQ3N2fJkiXkzZuXGzdu0L59e+rUqUNgYKDa0bK8snkcKeRsS1RsAltPhaod552kGFGBq6tr0s3R0RGNRvPKfYmcnJxwdXWlePHi/PHHH1hZWbFhw4akj+t0OlxdXXF3d6dWrVp88sknHD58mIi3jFn//zRNfHw8Q4YMwcnJiRw5cjB8+PBX/rrR6/X89NNPFCxYEGtra8qWLcvq1auTfY1evXpRoEABrK2tKVasWLKiSoiMkDevMnJy5owyavLhh8runDlzlDb0ffvC5s1w3zimzd/M3lUpSLLlV6ZqFn4EESFqpwLggw8+4OzZs3z//fdYW1uzb98+PD096devHzExxtUF1JRoNBpav+g5stoIpmpMrhjR6/VExsSpcsvo4Ulzc3N0Oh2xb2hbGRoaip+fH2ZmZpiZmaX4606ZMoV58+Yxd+5c9u/fz8OHD1m7dm2yx3z11VfMnz+fWbNmcebMGQYPHkznzp3Zs2cPAAkJCeTJk4eVK1cSHBzMN998w+jRo6VttMgUGo0yarJ1Kxw4oCxyjY1V1pZ89BE4OyvFSZcuypk5RjmT4JhbKUgc88LDK8oIyZO7aqcCwMbGhm+++Ybz58/ToUMH9Ho9s2bNwsfHhyg5kEg1rSvkRqOBo1cfcuOBYW9H06kdIL09j42n5Dd/q/LcwWMbY2ORMS9pdHQ0P//8MxERETRo0CDp/lOnTmFnZ0dCQgLPnz8HYMCAAdja2qb4a0+bNo1Ro0bh4+MDwB9//MHff//3Gj579oypU6eyc+dOqlWrBkDBggXZv38/f/75J3Xq1MHc3Jzvv/8+6XMKFCjAwYMHWbly5TunjYRIT9Wrwz//KFuGFy5UmqidPQuXLim3JUuUrrAzZ4Kbm9ppU8kpL3TfCPObwYOLSh+S7puVtSUGwMPDg2XLltGpUyfatGnDpk2baNGiBevWrcPGxkbteFmOm6M1NQrlZP+l+6wNvM3AhkXUjvRGJjcyYmo6dOiAnZ0dNjY2TJ06lcmTJ9OkSZOkjxcrVoygoCCOHTvG+PHjKVeuHOPHj0/x1w8PDyckJCSpyABl6qdixYpJ7wcHBxMVFUWjRo2ws7NLui1atIjLl//bbvjHH39QsWJFnJ2dsbOzY/bs2dIQSaimTh1ll01wsNLR9e+/YcgQ0OmU0ZGSJZXFsEY3SpItv7Ko1d4N7p1TOrVGPlQ7VTLNmjVjy5Yt2Nra4u/vT9OmTXnyxHi6gZoSH0+l54hf4C2DXlxsciMj1uZmBI9trNpzp7dffvmFhg0b4uDgQK5cuV75uIWFBYULFwaUxawXL17ks88+Y/HixemWIfH8ic2bN5M7d+5kH7O0tARg5cqVDB48mClTplCtWjXs7e35+eefOXLkSLrlECKtnJzAy0u5deum7Lo5cQJ69IAVK5RFsBUqqJ0yFXIUgm6bYEFTuHtaKUi6bQDrbGonS1KvXj3+/vtvmjRpwp49e/Dy8mLr1q04OTmpHS1LaVzKFVuL01x/EMnx64+olD+72pFey+RGRjQaDTYWOlVuGo0m3b8fV1dXChcu/NpC5HW+/vprli9fTkBAQIoe7+joiJubG4cPH066Ly4ujhMn/jvGvGTJklhaWnLjxg0KFy6c7ObxogHEvn37qF69Ov369aN8+fIULlw42aiJEIaiTBll6mbiRLC0VEZMPD2VkZS1ayE+Xu2EKZSzsLKGxNYZQk8qp/0+f6x2qmRq1KjBjh07yJYtG4cPH6ZFixZER0erHStLsbHQ0eQDZT5yzQnDXchqcsVIVlewYEG8vb2TmhGlxMCBA5k4cSJr167l3Llz9OvXj8ePHyd93N7enqFDhzJ48GAWLlzI5cuXCQwM5Pfff2fhwoUAFC5cmOPHj/P3339z4cIFvv76a44dO5be354Q6UKngxEjlIP6OnZU3t+7F1q3Vha6jhypTPHs3w937xrwVI5zMaVTq00OuBMIS3wgyrC6v1WqVImdO3fi4ODAvn376Nmzp0FPF5iixJN8N58MISrWMKttKUZM0JdffsnmzZtTPEXy5Zdf0rVrV7p37540xdKqVatkj/nhhx/45ptvmDBhAiVKlKBx48Zs3LiRAgUKANC3b19at25N+/btqVKlCg8ePKBfv37p/r0JkZ6KF4elS+HaNRg9WjlJ+OpVmDRJaaRWqxa4uirN1wYMgIsX1U78Gi4llU6t1tng9nFY2gaiDWt9Rrly5VizZg06nY5ly5bx7bffqh0pS6lSIDu5nax5Eh3H9mDD2IH1/zR6IyhRIyIicHR0JDw8HAcHh2Qfi4qK4urVqxQoUAArKyuVEor0ID9LobbISFi5Eo4fVwqPCxeU5mmJ/5fUaJStwoMGKW3oM2BmNu3uBCm7a6LCIW916LwaLFK+qy4zzJ07l969ewNKQ8fEbtMi403Zfp7pOy9Rp6gzC3tWzrTnfdvv75fJyIgQQrxgYwPdu8OMGcpakqtX4flz2L4dmjVTipKNG5U+JkWLKkWJvz8YxDII93LQZS1YOsCNg7CsPcQYVm+JXr16MWrUKAA++eQTdu7cqXKirCOxAdq+i/cIizC83i9SjAghxFtYWkKjRrBpE5w/D/37KycIX7qknBzs5aVM43h7w+TJyjqTSLVqgNye0NkPLOzh2j5Y0RFiDesXz7hx42jfvj1xcXG0atUqWUdpkXEK5LTFM182EvSwLsgwjhN4mRQjQgiRQkWLKqMmISHg56esK3Fzg2fPYMMGGDZMWWfi6Kjs0Bk9Gt5xbmX686ikTNGY28KVXeDbCeIMYehGodVqWbBgAfXq1SMiIgJvb29Gjx5NXFyc2tFMXusKSmuGNSduG9wiYilGhBAileztoVUr5Qyc27chIEBZ9NqqlVKcxMUp902YAAUKQPv28NL5lhkvb1XotBJ01nDpH1jZFeIM55wYKysrtm3bxsCBAwGYMGECXl5e3L1rmIsrTcVHZdyx0Gk5f/cJZ+4Y1q4rKUaEEOI9aDRQvjwMH66Mlty+DTduKLt06tVT+pasXKm0qa9cGb79VjlD52FGN03NXxM6+oLOCi5sg9U9IN5wjpK3sLBg2rRp+Pr6Ymtry65du6hQoQJLlixJOtpCpC9Ha3MalXQBYI2BHZ4nxYgQQqQjjQY8PJT+JTt3Kr1MevQACws4dgzGjoWmTZV1JsWLKx1hZ82CwEBlRCVdFawDHy8DM0s4twnW9IZ4w5oOadeuHceOHaNEiRLcuXOHLl264O7uTv/+/VPcvFGkXJsXC1k3BN0hNj5B5TT/ka29wmDIz1KYsrAwWL1a6f56+PDre5bY2EDFilC16n+3dDnM78J2Ze1IfAx80BZa/Qna9D++4n08ffqUX375hblz53L9+vWk+2vWrMnGjRuljXw6iYtPoOqEndx/Gs3srhWTRkoySkq39koxIgyG/CxFVvLgARw5otwOH1behoe/+jgPD+XcnCJFlAW0iW/d3FLZ5+T8VvDtDAlxULYDeP9ucAUJKGdh7dy5k7lz5+Ln50dMTAxt2rRh5cqVGXLkRlY0blMwc/ZfpUlpV2Z19szQ55JiRBgd+VmKrCwhQdk6fOjQfwXK6dPK/a/j5pZ8BKVUKeVAQLO31RfBG2BVd9DHQ/nO0Hw6aA13tv7o0aPUqFGDuLg4Zs2aRd++fdWOZBLOhkTQ5Nd9WJhpOTqmAU42Fhn2XNL0TLzWtWvX0Gg0BAUFvfExu3fvRqPRJDuf5n1lxNcUwpRotVCihHKi8J9/wr//wuPHyrqT6dOVdvRNmkDhwkrBERKiHOw3YoRyyF/OnGBurqxFKVIEatSAH35QGrclKdkCfOaARguBS2DzEAM+eAcqV67MxIkTARg0aBAnT55UOZFpKOHmQEk3B2LiE9j47x214wBSjKime/fuaDSaV26XLl165ePm5uYULFiQoUOH8uzZM+C/oiLx5ujoSNWqVdm4ceNbn9fDw4OQkBBKly6d4d+jEOL92NsrO3I+/1xpsLZli7LWJCIC9u2Dn38GHx/IrbSPQK9XdulcugQHD8I330DBglC7NsyerRQ3lG4Nrf5SCpIT82HLMIMuSAYPHkyzZs2Ijo6mXbt2PH36VO1IJiGp50iAYTRAk2JERR9++CEhISHJbokHz7388StXrjBu3DhmzpzJ0KFDk32Nf/75h5CQEI4cOULlypXx8fHh9OnTb3xOMzMzXF1d0el0GfZ9CSEylo0N1KwJQ4cqi2Jv3VJa0oeGwpkzygnEc+cqbes1GqVw+fRTpWjp1w/OWbQF75mABo7Nhm2jDLYgSWySljt3bs6fP8/nn3+udiST4F0uN2ZaDUE3H3P5nvoFnhQjKrK0tMTV1TXZzeylCd/Ej3t4eNCxY0c6derEunXrkn2NHDly4OrqSvHixRk/fjyxsbHs2rXrjc/5ummaLVu2ULRoUaytralXrx7Xrl175fMOHjxI7dq1sba2xsPDgwEDBiSN0gAsWbKEihUrYm9vj6urKx07diQsLCzNr40QInUsLMDFBUqWVLrA9uwJ//yj9DyZNElZUxIZqWwjLlECmo7uwOlC05VPPjIL/L822IIkZ86cLFu2DK1Wy8KFC2nTpg1Tpkxh165dPHr0SO14RsnZ3pK6RZ0BWHNC/Z4jpleM6PUQ80ydWwb/Q7a2tiY29vVNi2JjY5k9ezYA5ubmKf6aN2/epHXr1jRt2pSgoCB69+7NyJEjkz3m1KlTNG7cmNatW3Py5El8fX3Zv39/sr9QYmJi+OGHH/j3339Zt24dV69elRM5hTAAefIoDdlOnYJdu5QzdDQapfHaB1278Pm2X5QHHpzOuZljuXZVT3y8uplfp3bt2owdOxaANWvWMHToUOrXr0/27NkpWbIkEydOJCQkROWUxiXx8Ly1gbeJT1C3EE3TWP3MmTP5+eefCQkJoVSpUkybNo1atWq98fF79uxhyJAhnDlzBnd3d4YPH55xq6JjI+FH94z52u8y+k6qjuzetGkTdnZ2Se83adKEVatWvfaxR48eZdmyZTRo0CDZ/dWrV0er1fL8+XMSEhLInz8/7dq1S3GGWbNmUbBgQX755Rc0Gg3FihXj1KlTTJo0KekxP//8Mx07dmTQoEEAFClShN9++406deowa9YsrKys6NmzZ9LjCxYsyG+//UblypV5+vRpsu9RCKEOjQbq1lVuly8ri2IXLYLfj/REnxDP702HUvzeVMb31vDD/q8oWEhL0aLKglkXF2VhbPbsytvEW7ZsyohMZhk9ejQ1atRg//79BAYGEhgYyNWrVzl79iyjRo3iq6++omnTpvTs2ZOPPvpIpqPfoUGJXDhY6QgJj+LwlQfUKJxTtSyp/kn5+voyaNAgZs6cSY0aNfjzzz9p0qQJwcHB5M2b95XHX716laZNm/LJJ5+wZMkSDhw4QL9+/XB2dsbHxyddvgljVa9ePWbNmpX0vq1t8kImsViJi4sjNjYWb29vpk+fnuwxvr6+FC9enAsXLjBo0CD++OMPsmfPnuIMZ8+epWrVqsn271erVi3ZY06cOMGlS5dYunRp0n16vZ6EhASuXr1KiRIlCAwM5LvvviMoKIiHDx+S8GI/4o0bNyhZsmSK8wghMl6hQjBtGkydChcuwOHDn7DyZCztHEYxptYU6uQ7QK8NM1h/tsg7v5a9ffIiJfH6/wuXHDmUc3qcnVPZH+UlGo2GunXrUrdu3aT7Hj58yLp165g3bx4HDhxg48aNbNy4MekP5YYNG6btybIAK3Mzmpd1Z+mRG6w5ccu4ipGpU6fSq1cvevfuDcC0adP4+++/mTVrFhMmTHjl8X/88Qd58+Zl2rRpAJQoUYLjx48zefLkjClGzG2UEQo1mNuk6uG2trYULlz4jR9PLFbMzc1xd3d/7fSLh4cHRYoUoUiRItjZ2eHj40NwcDC5cuVKUYaUtJlJSEigT58+DBgw4JWP5c2bl2fPnuHl5YWXlxdLlizB2dmZGzdu0LhxY2JiDOdwLiFEclqt0pK+eHGAfhBgj37bSGrmPUzwFzU4bD0av5DPCbuv4+FDpVHbgwfKjp1Hj5SZ6SdPlNtLTVPfysFB2XpcpAjkz/9qEZN4nT27slX5XbJnz07Pnj3p2bMn586dY968ecydO5czZ87QqFEjvL29mTJlCoUKFXqPV8p0+XjmYemRG2w9HcrYlnHYWaozmpSqZ42JieHEiROvrCnw8vLi4MGDr/2cQ4cO4eXlley+xo0bM3fuXGJjY1/7CzY6Opro6P+OvI6ISMXpghpNqqZKDNm7ipX/V6dOHUqXLs348eP59ddfU/Q5JUuWfGVR7OHDh5O9X6FCBc6cOfPGLKdOneL+/ftMnDgRDw8PAI4fP57i3EIIA1GhC5qCdWDjIMwu76BG5LfUKL4W8lZ75aEJemUHT1TU/92ew/P/vy8Knj+Hp08g6c+fRy9uQPiL29X/ew4LC7Cy+u9mbQWWVm9v7NbMTEvjnj0IDQ3l/r176KOes+nz/mTLli3ZBgHxn2/dY9Hr4a9ZDxgyqJMqGVJVjNy/f5/4+HhcXJL3sndxcSE0NPS1nxMaGvrax8fFxXH//n3cXnPwwoQJE/j+++9TE0288OWXX9K2bVuGDx9O7sTmA2/Rt29fpkyZwpAhQ+jTpw8nTpxgwYIFyR4zYsQIqlatSv/+/fnkk0+wtbXl7Nmz+Pv7M336dPLmzYuFhQXTp0+nb9++nD59mh9++CGDvkMhRIZyygud10DQMvh7FIQEKbf/owWsX9xe+0GbFzc1WAKFXtxEiv1++wPVnjtN4zH/fz6AXq9/65kBr3v86+5PNGrUKIYMGZL0fkRERNJf3OLtPvroI/Lnz8/48eOZOXPmOx+fN29e1qxZw+DBg5k5cyaVK1fmxx9/TLYgtUyZMuzZs4cxY8ZQq1Yt9Ho9hQoVon379gA4OzuzYMECRo8ezW+//UaFChWYPHkyLVq0yLDvUwiRgTQaKN8JCtWHoKXKxoBMlJAAUdHKaErUc+Vt5PP/Rl30qThsVg88fvQodSPsWVRRr/KqPXeqzqaJiYnBxsaGVatW0apVq6T7Bw4cSFBQEHv27Hnlc2rXrk358uWTTRusXbuWdu3aERkZmaJtqHI2TdYgP0shhDAtGXI2jYWFBZ6envj7+ye739/fn+rVq7/2c6pVq/bK47dv307FihVT1Q9DCCGEEKYp1U3PhgwZwpw5c5g3bx5nz55l8ODB3LhxI6lvyKhRo+jatWvS4/v27cv169cZMmQIZ8+eTVrp/P9tzYUQQgiRNaV6zUj79u158OABY8eOTTpwbcuWLeTLlw+AkJAQbty4kfT4AgUKsGXLFgYPHszvv/+Ou7s7v/32W5bvMSKEEEIIRarWjKhF1oxkDfKzFEII05Iha0aEEEIIIdKbyRQjie3HhfGSn6EQQmRNRn+KkIWFBVqtljt37uDs7IyFhcVbe54Iw6PX64mJieHevXtotVosMvPkLSGEEKoz+mJEq9VSoEABQkJCuHNHpTNpRLqwsbEhb968aLUmM2AnhBAiBYy+GAFldCRv3rzExcURHx+vdhyRBmZmZuh0OhnVEkKILMgkihFQWsubm5tLIzUhhBDCyMh4uBBCCCFUJcWIEEIIIVQlxYgQQgghVGUUa0YSm8TKEdBCCCGE8Uj8vf2uZu9GUYw8efIEAA8PD5WTCCGEECK1njx5gqOj4xs/bhRn0yQkJHDnzh3s7e3TdetnREQEHh4e3Lx5860988X7k9c6c8jrnDnkdc4c8jpnjox8nfV6PU+ePMHd3f2tPaSMYmREq9WSJ0+eDPv6Dg4O8h96JpHXOnPI65w55HXOHPI6Z46Mep3fNiKSSBawCiGEEEJVUowIIYQQQlVZuhixtLTk22+/xdLSUu0oJk9e68whr3PmkNc5c8jrnDkM4XU2igWsQgghhDBdWXpkRAghhBDqk2JECCGEEKqSYkQIIYQQqpJiRAghhBCqMvliZObMmRQoUAArKys8PT3Zt2/fWx+/Z88ePD09sbKyomDBgvzxxx+ZlNS4peZ19vPzo1GjRjg7O+Pg4EC1atX4+++/MzGtcUvtf9OJDhw4gE6no1y5chkb0ESk9nWOjo5mzJgx5MuXD0tLSwoVKsS8efMyKa3xSu3rvHTpUsqWLYuNjQ1ubm706NGDBw8eZFJa47R3716aN2+Ou7s7Go2GdevWvfNzMv13od6ErVixQm9ubq6fPXu2Pjg4WD9w4EC9ra2t/vr16699/JUrV/Q2Njb6gQMH6oODg/WzZ8/Wm5ub61evXp3JyY1Lal/ngQMH6idNmqQ/evSo/sKFC/pRo0bpzc3N9QEBAZmc3Pik9rVO9PjxY33BggX1Xl5e+rJly2ZOWCOWlte5RYsW+ipVquj9/f31V69e1R85ckR/4MCBTExtfFL7Ou/bt0+v1Wr1v/76q/7KlSv6ffv26UuVKqVv2bJlJic3Llu2bNGPGTNGv2bNGj2gX7t27Vsfr8bvQpMuRipXrqzv27dvsvuKFy+uHzly5GsfP3z4cH3x4sX/1979hETRxnEA/66OSyIoaGSLC0agogehWrRWRAhRSOgkCYooJCQRklLgQVIhCIo8KNZJ9KIiCoIHFT2kKHpRVhA3KKwOi1lsEWxa5J/fewh939V933bmbWaa8fuBOfj4LPzmy+jz22dn2LCxW7duyeXLl3Wr0Q7U5hxJTk6OtLe3/+7SbEdr1hUVFdLS0iKtra1sRqKgNueJiQlJSkqST58+GVGebajN+cmTJ3L+/Pmwsc7OTnG73brVaDfRNCNmrIW2/Zjmx48fWF5eRklJSdh4SUkJFhYWIr5mcXHx2PzS0lIsLS1hZ2dHt1qtTEvOR+3v7yMUCiE5OVmPEm1Da9a9vb1YX19Ha2ur3iXagpacx8bG4PF48PjxY6SlpSEzMxP37t3Dt2/fjCjZkrTk7PV6EQgEMD4+DhHBhw8fMDIygrKyMiNKPjHMWAst8UV5WgSDQezt7SE1NTVsPDU1FZubmxFfs7m5GXH+7u4ugsEgXC6XbvValZacj3r69Cm2trZw48YNPUq0DS1Zv379Gs3NzZibm4Oi2PbP/bfSkvObN28wPz+PU6dOYXR0FMFgELdv38bnz59538i/0JKz1+tFf38/Kioq8P37d+zu7uL69evo6uoyouQTw4y10LY7IwccDkfYzyJybOxX8yONUzi1OR8YHBxEW1sbhoaGcObMGb3Ks5Vos97b20NlZSXa29uRmZlpVHm2oeaa3t/fh8PhQH9/P/Ly8nDt2jV0dHSgr6+PuyO/oCZnv9+PhoYGPHjwAMvLy5icnMTbt29RX19vRKknitFroW3fKp0+fRqxsbHHOuyPHz8e6/gOnD17NuJ8RVGQkpKiW61WpiXnA0NDQ7h58yaGh4dRXFysZ5m2oDbrUCiEpaUl+Hw+3LlzB8DPRVNEoCgKpqamcPXqVUNqtxIt17TL5UJaWlrYV6VnZ2dDRBAIBJCRkaFrzVakJedHjx6hoKAA9+/fBwDk5uYiISEBhYWFePjwIXevfxMz1kLb7ow4nU5cunQJ09PTYePT09Pwer0RX3PlypVj86empuDxeBAXF6dbrVamJWfg545IbW0tBgYG+HlvlNRmnZiYiNXVVaysrBwe9fX1yMrKwsrKCvLz840q3VK0XNMFBQXY2NjA169fD8devXqFmJgYuN1uXeu1Ki05b29vIyYmfNmKjY0F8Pc7d/r/TFkLdbs19g9w8NhYT0+P+P1+uXv3riQkJMi7d+9ERKS5uVmqq6sP5x88ztTY2Ch+v196enr4aG8U1OY8MDAgiqJId3e3vH///vD48uWLWadgGWqzPopP00RHbc6hUEjcbreUl5fL2tqazM7OSkZGhtTV1Zl1CpagNufe3l5RFEWePXsm6+vrMj8/Lx6PR/Ly8sw6BUsIhULi8/nE5/MJAOno6BCfz3f4CPWfsBbauhkREenu7pb09HRxOp1y8eJFmZ2dPfxdTU2NFBUVhc2fmZmRCxcuiNPplHPnzsnz588Nrtia1ORcVFQkAI4dNTU1xhduQWqv6X9iMxI9tTm/fPlSiouLJT4+XtxutzQ1Ncn29rbBVVuP2pw7OzslJydH4uPjxeVySVVVlQQCAYOrtpYXL1785//cP2EtdIhwb4uIiIjMY9t7RoiIiMga2IwQERGRqdiMEBERkanYjBAREZGp2IwQERGRqdiMEBERkanYjBAREZGp2IwQERGRqdiMEBERkanYjBAREZGp2IwQERGRqdiMEBERkan+AnKNnxqi4sqSAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR', color='black')\n",
    "plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR', color='blue')\n",
    "\n",
    "plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR ideal')\n",
    "plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR ideal')\n",
    "#plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR random', color='grey')\n",
    "#plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR random', color='grey')\n",
    "\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d3e84a4-7c93-48db-aabc-e5efaa1af871",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7170f58b-1924-49cb-8122-b2f8295136d8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e5ee37c3-071e-4094-ad82-25ff79392499",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "bc605391",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x1330b5ed0>"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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2jT27NnDvHnzmDt3Lg899JBVP4KIiNRyls71Onr0aJKTk3niiSeIj4+na9eurFixgpYtWwIQHx9f6prKmJgYVqxYwYMPPsjrr79O06ZNeeWVV7jhhhus+hFERKSWs/Q6SivUxHWUIiLi+dz+OkoRERFPoKAUERGpgIJSRESkAgpKERGRCigoRUREKqCgFBERqYCCUkREpAIKShERkQooKEVERCqgoBQREamApXO9WqFoxr60tDSLKxERESsV5cAfzeRa54IyPT0dgOjoaIsrERERd5Cenk5YWFi5r9e5SdEdDgcnTpwgJCQEm8123u+TlpZGdHQ0cXFxmlz9LDou5dOxKZuOS/l0bMpWU8fFMAzS09Np2rQpXl7ln4mscy1KLy8vmjdvXmPvFxoaqn/AZdBxKZ+OTdl0XMqnY1O2mjguFbUki2gwj4iISAUUlCIiIhVQUJ4nf39/Hn/8cfz9/a0uxa3ouJRPx6ZsOi7l07Epm6uPS50bzCMiIlIValGKiIhUQEEpIiJSAQWliIhIBRSUIiIiFVBQVmD27NnExMQQEBBAbGws69evr3D7devWERsbS0BAAK1bt+aNN95wUaWuVZXj8sknn3DFFVfQuHFjQkND6devH6tWrXJhta5V1X8zRb777jt8fHzo0aOHcwu0SFWPS25uLo8++igtW7bE39+fNm3aMG/ePBdV61pVPTaLFi2ie/fu1KtXj6ioKCZNmkRycrKLqnWNb7/9lhEjRtC0aVNsNhvLly//w32c+v1rSJk++OADw9fX13j77beN3bt3Gw888IARFBRkHDlypMztDx48aNSrV8944IEHjN27dxtvv/224evra3z88ccurty5qnpcHnjgAeOZZ54xNm/ebOzbt8+YMWOG4evra2zbts3FlTtfVY9NkZSUFKN169bGkCFDjO7du7umWBc6n+NyzTXXGH369DFWr15tHDp0yNi0aZPx3XffubBq16jqsVm/fr3h5eVlvPzyy8bBgweN9evXG126dDFGjhzp4sqda8WKFcajjz5qLFu2zACMTz/9tMLtnf39q6AsR+/evY0pU6aUWtexY0fjkUceKXP7hx9+2OjYsWOpdXfeeafRt29fp9Vohaoel7J07tzZmDlzZk2XZrnzPTajR482/v73vxuPP/54rQzKqh6Xr776yggLCzOSk5NdUZ6lqnpsnnvu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",
      "text/plain": [
       "<Figure size 500x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(5, 5))\n",
    "\n",
    "plt.plot(df_scores.fpr, df_scores.tpr, label='Model')\n",
    "plt.plot([0, 1], [0, 1], label='Random', linestyle='--')\n",
    "\n",
    "plt.xlabel('FPR')\n",
    "plt.ylabel('TPR')\n",
    "\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "4a689523",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import roc_curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "37c94946",
   "metadata": {},
   "outputs": [],
   "source": [
    "fpr, tpr, thresholds = roc_curve(y_val, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "5863648f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x1331792d0>"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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RMQO3aMUERGxQ0EpIiJih4JSRETEDgWliIiIHQpKEREROxSUIiIidigoRURE7FBQioiI2KGgFPEQo0aNwsfHp9zj119/LfVcQEAALVu25NFHHyU7OxuA3377rdRrIiIi6NatGx999JHJ70rEfApKEQ9y3XXXkZaWVuoRExNT6rm9e/fy9NNPM3v2bB599NFSr//ss89IS0vjm2++oUuXLtx66638+OOPZrwVEZehoBTxIEFBQTRu3LjUo2h1iaLnoqOjGTZsGHfeeSerV68u9foGDRrQuHFj2rZtyz//+U/y8/NZt26dCe9ExHUoKEW8VEhICPn5+RU+l5+fz5tvvgngMcvEidSUFm4W8SBr1qyhTp06tu0BAwawfPnycsd9++23LFmyhGuuuabU/h49euDr68vp06exWCy0aNGCoUOHOrxuEVemoBTxIH379mXOnDm27bCwMNvXRSFaUFBAfn4+N910E6+88kqp1ycnJ9O2bVt2797NxIkTmTt3LvXr13da/SKuSEEp4kHCwsJo3bp1hc8VhWhAQABNmjSp8JJqdHQ0bdq0oU2bNtSpU4dbb72VnTt30qhRI0eXLuKydI9SxEsUhWjz5s2rdN+xT58+dOzYkX/+859OqE7EdSkoRaRSjzzyCK+//jqHDx82uxQR0ygoRaRSN9xwAy1atFCrUryaj2EYhtlFiIiIuCq1KEVEROxQUIqIiNihoBQREbFDQSkiImKHglJERMQOBaWIiIgdCkoRERE7FJQiIiJ2KChFRETsUFCKiIjYoaAUERGxQ0EpIiJix/8DkBrCXptyiqkAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 500x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(5, 5))\n",
    "\n",
    "plt.plot(fpr, tpr, label='Model')\n",
    "plt.plot([0, 1], [0, 1], label='Random', linestyle='--')\n",
    "\n",
    "plt.xlabel('FPR')\n",
    "plt.ylabel('TPR')\n",
    "\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "886f8028",
   "metadata": {},
   "source": [
    "## 4.6 ROC AUC\n",
    "\n",
    "* Area under the ROC curve - useful metric\n",
    "* Interpretation of AUC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "7509a4b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import auc #for any curve "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "6643a6e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8438555705812935"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "auc(fpr, tpr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "e1c0c11b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8438796286447966"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "auc(df_scores.fpr, df_scores.tpr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "0774697c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9999430203759136"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "auc(df_ideal.fpr, df_ideal.tpr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "9f8658f5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8438555705812935"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fpr, tpr, thresholds = roc_curve(y_val, y_pred)\n",
    "auc(fpr, tpr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "29cbdd64",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import roc_auc_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "9fd31898",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8438555705812935"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "roc_auc_score(y_val, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "02bb73e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "neg = y_pred[y_val == 0]\n",
    "pos = y_pred[y_val == 1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "cfeb22b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "ee7da8e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.84432"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n = 100000\n",
    "success = 0 \n",
    "\n",
    "for i in range(n):\n",
    "    pos_ind = random.randint(0, len(pos) - 1)\n",
    "    neg_ind = random.randint(0, len(neg) - 1)\n",
    "\n",
    "    if pos[pos_ind] > neg[neg_ind]:\n",
    "        success = success + 1\n",
    "\n",
    "success / n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "2b9e19e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.84646"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n = 50000\n",
    "\n",
    "np.random.seed(1)\n",
    "pos_ind = np.random.randint(0, len(pos), size=n)\n",
    "neg_ind = np.random.randint(0, len(neg), size=n)\n",
    "\n",
    "(pos[pos_ind] > neg[neg_ind]).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6336a636",
   "metadata": {},
   "source": [
    "## 4.7 Cross-Validation\n",
    "\n",
    "* Evaluating the same model on different subsets of data\n",
    "* Getting the average prediction and the spread within predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "92708443",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(df_train, y_train, C=1.0):\n",
    "    dicts = df_train[categorical + numerical].to_dict(orient='records')\n",
    "\n",
    "    dv = DictVectorizer(sparse=False)\n",
    "    X_train = dv.fit_transform(dicts)\n",
    "\n",
    "    model = LogisticRegression(C=C, max_iter=1000)\n",
    "    model.fit(X_train, y_train)\n",
    "    \n",
    "    return dv, model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "4689044d",
   "metadata": {},
   "outputs": [],
   "source": [
    "dv, model = train(df_train, y_train, C=0.001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "08ac302c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(df, dv, model):\n",
    "    dicts = df[categorical + numerical].to_dict(orient='records')\n",
    "\n",
    "    X = dv.transform(dicts)\n",
    "    y_pred = model.predict_proba(X)[:, 1]\n",
    "\n",
    "    return y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "95d88ee1",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = predict(df_val, dv, model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "e3c7e607",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import KFold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e1d8a427",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "d5c8e07d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: tqdm in /opt/anaconda3/lib/python3.11/site-packages (4.65.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "a8fe7363",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tqdm.auto import tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "ce936aca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8632f977d73a46be8af70cff436d730b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/7 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C=0.001 0.825 +- 0.009\n",
      "C=0.01 0.840 +- 0.009\n",
      "C=0.1 0.841 +- 0.008\n",
      "C=0.5 0.841 +- 0.007\n",
      "C=1 0.840 +- 0.009\n",
      "C=5 0.840 +- 0.008\n",
      "C=10 0.842 +- 0.007\n"
     ]
    }
   ],
   "source": [
    "n_splits = 5\n",
    "\n",
    "for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):\n",
    "    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)\n",
    "\n",
    "    scores = []\n",
    "\n",
    "    for train_idx, val_idx in kfold.split(df_full_train):\n",
    "        df_train = df_full_train.iloc[train_idx]\n",
    "        df_val = df_full_train.iloc[val_idx]\n",
    "\n",
    "        y_train = df_train.churn.values\n",
    "        y_val = df_val.churn.values\n",
    "\n",
    "        dv, model = train(df_train, y_train, C=C)\n",
    "        y_pred = predict(df_val, dv, model)\n",
    "\n",
    "        auc = roc_auc_score(y_val, y_pred)\n",
    "        scores.append(auc)\n",
    "\n",
    "    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "2f72b194",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.8438154971701297,\n",
       " 0.8458585783339954,\n",
       " 0.8333253060405378,\n",
       " 0.8337776587580739,\n",
       " 0.8526756298829656]"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "03e81326",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8572386167896259"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#testing model\n",
    "dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)\n",
    "y_pred = predict(df_test, dv, model)\n",
    "\n",
    "auc = roc_auc_score(y_test, y_pred)\n",
    "auc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c30a306d-1591-4143-a315-07ab2216bc46",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "ddb116c3-4690-45a6-a2fd-20ef812969ca",
   "metadata": {},
   "source": [
    "saving model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "febdeffb-3d5a-4c1b-9b2e-8989c7db6767",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "dc8b2933-0036-4415-8599-48db9f533e44",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'model_C=10.bin'"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "output_file = f'model_C={C}.bin'\n",
    "output_file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "7d13eb95-6562-4a30-af3f-5cd228092aa8",
   "metadata": {},
   "outputs": [],
   "source": [
    "f_out = open(output_file, 'wb')\n",
    "pickle.dump((dv,model),f_out)\n",
    "f_out.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "ec7bfe7d-11f1-4220-b921-ff36859d1c9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(output_file, 'wb') as f_out:\n",
    "    pickle.dump((dv, model), f_out\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dbbd20f0-006b-46de-95bd-c3c2a7c8a18e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#load model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "9628fe94-6d41-440f-9f75-ff48a60acbdf",
   "metadata": {},
   "outputs": [],
   "source": [
    "output_file = 'model_C=10.bin'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "36a39878-72fc-4832-ad41-8dc5d1dc865f",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(output_file, 'rb') as f_in:\n",
    "    dv, model =pickle.load(f_in)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "933cd3ec-96c2-4a00-8120-8c75358c50b7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(DictVectorizer(sparse=False), LogisticRegression(max_iter=1000))"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dv, model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "92ff62dc-0a7a-445a-afd0-9a75c1388623",
   "metadata": {},
   "outputs": [],
   "source": [
    "customer={'gender':'Male', 'seniorcitizen':'0', 'partner':'Yes', 'dependents':'Yes', 'tenure':'12', 'phoneservice':'Yes', 'multiplelines':'No', 'internetservice':'No', 'onlinesecurity':'No internet service', 'onlinebackup':'No internet service', 'deviceprotection':'No internet service', 'techsupport':'No internet service', 'streamingtv':'No internet service', 'streamingmovies':'No internet service', 'contract':'Two year', 'paperlessbilling':'No', 'paymentmethod':'Mailed check', 'monthlycharges':'19.7', 'totalcharges':'258.35', 'churn':'0' }\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "86df78bc-8a7e-4d7a-ae35-a88bd0cc9c07",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dv.transform([customer])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "4732bd3b-95ab-4434-8b1c-3602e48263c6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0.,\n",
       "        0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0.,\n",
       "        1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0.]])"
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "78ecd507-7050-4023-92d8-857afb3e439f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.10706364413055769"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict_proba(X)[0,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4ec65e1f-10ba-4e77-97aa-22d72bd9a6e4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f37b6ea-09ba-4769-96eb-47417c5d9263",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2a1d5e5e-4453-42dc-8159-e10afc39e52b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a7ccbe2-462b-42ae-bed2-b599fbee6c5f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4d563b62-11b8-4b6e-afad-8287e276a03f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "db242dff",
   "metadata": {},
   "source": [
    "## 4.8 Summary\n",
    "\n",
    "* Metric - a single number that describes the performance of a model\n",
    "* Accuracy - fraction of correct answers; sometimes misleading \n",
    "* Precision and recall are less misleading when we have class inbalance\n",
    "* ROC Curve - a way to evaluate the performance at all thresholds; okay to use with imbalance\n",
    "* K-Fold CV - more reliable estimate for performance (mean + std)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4941b0ca",
   "metadata": {},
   "source": [
    "## 4.9 Explore more\n",
    "\n",
    "* Check the precision and recall of the dummy classifier that always predict \"FALSE\"\n",
    "* F1 score = 2 * P * R / (P + R)\n",
    "* Evaluate precision and recall at different thresholds, plot P vs R - this way you'll get the precision/recall curve (similar to ROC curve)\n",
    "* Area under the PR curve is also a useful metric\n",
    "\n",
    "Other projects:\n",
    "\n",
    "* Calculate the metrics for datasets from the previous week"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ade60b8d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
