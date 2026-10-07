from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib


# ============================================================
# LOAD MODEL AND PREPROCESSING FILES
# ============================================================

model = joblib.load("xgb_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")
education_mapping = joblib.load("education_mapping.pkl")
selected_features = joblib.load("selected_features.pkl")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Credit Risk Classification using XGBoost",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INPUT DATA MODEL
# ============================================================

class CreditRiskInput(BaseModel):

    Total_TL_opened_L6M: float
    Tot_TL_closed_L6M: float
    pct_tl_open_L6M: float
    pct_tl_closed_L6M: float
    pct_tl_open_L12M: float
    pct_tl_closed_L12M: float

    Tot_Missed_Pmnt: float
    CC_TL: float
    Consumer_TL: float
    Gold_TL: float
    Home_TL: float
    PL_TL: float
    Other_TL: float

    Age_Oldest_TL: float
    Age_Newest_TL: float

    time_since_recent_payment: float
    num_times_delinquent: float
    max_recent_level_of_deliq: float
    num_deliq_6_12mts: float
    num_times_60p_dpd: float

    num_std: float
    num_std_6mts: float
    num_sub: float
    num_sub_6mts: float
    num_sub_12mts: float
    num_dbt: float
    num_dbt_6mts: float
    num_lss: float

    recent_level_of_deliq: float

    tot_enq: float
    CC_enq: float
    CC_enq_L6m: float
    PL_enq: float
    PL_enq_L6m: float

    time_since_recent_enq: float
    enq_L3m: float

    AGE: float
    NETMONTHLYINCOME: float
    Time_With_Curr_Empr: float

    pct_of_active_TLs_ever: float
    pct_opened_TLs_L6m_of_L12m: float

    CC_Flag: float
    PL_Flag: float

    pct_PL_enq_L6m_of_ever: float
    pct_CC_enq_L6m_of_ever: float

    HL_Flag: float
    GL_Flag: float

    Credit_Score: float

    EDUCATION: str
    MARITALSTATUS: str
    GENDER: str

    last_prod_enq2: str
    first_prod_enq2: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Credit Risk Prediction API is running",
        "status": "success",
        "docs": "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "features": len(feature_columns)
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: CreditRiskInput):

    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    input_data = data.model_dump()

    df = pd.DataFrame([input_data])


    # --------------------------------------------------------
    # EDUCATION MAPPING
    # --------------------------------------------------------

    if "EDUCATION" in df.columns:

        df["EDUCATION"] = (
            df["EDUCATION"]
            .map(education_mapping)
            .fillna(0)
        )


    # --------------------------------------------------------
    # ONE-HOT ENCODING
    # --------------------------------------------------------

    categorical_columns = [
        "MARITALSTATUS",
        "GENDER",
        "last_prod_enq2",
        "first_prod_enq2"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )


    # --------------------------------------------------------
    # ADD MISSING TRAINING FEATURES
    # --------------------------------------------------------

    for column in feature_columns:

        if column not in df.columns:

            df[column] = 0


    # --------------------------------------------------------
    # KEEP EXACT TRAINING FEATURE ORDER
    # --------------------------------------------------------

    df = df[feature_columns]


    # --------------------------------------------------------
    # CONVERT TO FLOAT
    # --------------------------------------------------------

    df = df.astype(float)


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction_encoded = model.predict(df)[0]

    prediction_probability = model.predict_proba(df)[0]


    # --------------------------------------------------------
    # DECODE PREDICTION
    # --------------------------------------------------------

    prediction_label = label_encoder.inverse_transform(
        [prediction_encoded]
    )[0]


    # --------------------------------------------------------
    # PROBABILITY DICTIONARY
    # --------------------------------------------------------

    probabilities = {}

    for class_value, probability in zip(
        label_encoder.classes_,
        prediction_probability
    ):

        probabilities[str(class_value)] = round(
            float(probability),
            4
        )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "status": "success",

        "prediction": str(prediction_label),

        "risk_category": str(prediction_label),

        "probabilities": probabilities

    }