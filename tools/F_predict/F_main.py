# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:14:48 2024

@author: jspark
"""
import pandas as pd
from .F_preprocess import smiles_to_mf, scaling_mf, preprocessing
from .F_predict import load_model, make_prediction
from .F_utils import pred_to_function, pred_proba_to_df, prediction_to_dataframe, product_category_matching, fill_max_function
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import os
# dir_path : script 파일 위치
# input_path : 예측할 SMILES list 파일이름 (csv or xlsx)
# output_path : output 파일 이름 (csv or xlsx)
# save_proba : True=예측 raw 결과 함께 출력 / False=기능명만 출력
def F_main(smiles_list):
    model_path = f'./tools/F_predict/parameters/Functions_weights.pth'
    scaler_path = f'./tools/F_predict/parameters/scaler.sav'
    dt = pd.DataFrame({'SMILES':smiles_list})
    
    mf = smiles_to_mf(smiles_list)
    mf = scaling_mf(mf, scaler_path)
    
    # smiles_list와 molecular feature로 예측 데이터셋 생성
    dataloader = preprocessing(smiles_list, mf)
    
    # 모델 파라미터 로드, 예측
    model = load_model(model_path)
    pred_proba, pred = make_prediction(model, dataloader)
    
    pred = fill_max_function(pred, pred_proba)
    
    # 예측 결과 -> 데이터프레임 정리
    df = prediction_to_dataframe(smiles_list, pred, pred_proba, False)
    df = product_category_matching(df, './tools/F_predict/' )
    
    dt['Functions'] = None
    dt.loc[dt['SMILES'].notna(), 'Functions'] = df['Functions'].values
    
    dt['Product Family'] = None
    dt.loc[dt['SMILES'].notna(), 'Product Family'] = df['Product Family'].values
    
       
    return list(dt['Functions'])


