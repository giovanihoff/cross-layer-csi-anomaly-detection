# Dataset Research

## IEEE-CIS Fraud Detection

- UID adotado: `card1_addr1 + floor(TransactionDT/86400 - D1)`.
- `TransactionID` fica preservado como `event_id`.
- Racional: a construcao aparece nas discussoes de solucao vencedora do Kaggle e em espelhos tecnicos usados pela comunidade.
- Preparo implementado nesta etapa: merge de `train_transaction` com `train_identity`, derivacao do UID, normalizacao dos campos `D*`, features temporais e agregacoes basicas por `uid` e `card1`.

Fontes:

- https://www.kaggle.com/competitions/ieee-fraud-detection/overview
- https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600
- https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284
- https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308
- https://github.com/zehuichen123/blog/blob/67d8286f9ca254e4a60d572b708030d4496af9d7/_posts/2019-10-13-relearning-tabular-data-competition-kaggle-ieee-fraud-detection.md
- https://github.com/amazon-science/fraud-dataset-benchmark/blob/main/src/fdb/preprocessing.py

## Sparkov

- UID adotado: `cc_num`.
- `trans_num` fica preservado como `event_id`.
- Racional: nao ha competicao Kaggle oficial com vencedor canonico para essa base; o projeto evita UID artificial e usa os identificadores nativos do dataset.
- Preparo implementado nesta etapa: preservacao do split original, parsing temporal, idade na transacao, distancia geografica e agregacoes por cartao e merchant.

Fontes:

- https://www.kaggle.com/datasets/kartik2112/fraud-detection
- https://github.com/amazon-science/fraud-dataset-benchmark
- https://github.com/amazon-science/fraud-dataset-benchmark/blob/main/src/fdb/preprocessing.py

## Fraud E-commerce

- UID adotado: `device_id` como identidade operacional principal.
- `user_id` fica preservado como `account_id`.
- `event_id` e criado de forma deterministica porque a fonte nao traz um transaction id explicito.
- Racional: tambem nao ha competicao Kaggle oficial com vencedor canonico; o uso de `device_id` evita sintetizar uma identidade ficticia e segue a pratica recorrente de engenharia de risco por dispositivo/IP nessa base.
- Preparo implementado nesta etapa: parsing temporal, `time_since_signup`, contagens por dispositivo/IP/usuario e enriquecimento opcional por pais via tabela de IP.

Fontes:

- https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce
- https://github.com/amazon-science/fraud-dataset-benchmark/blob/main/src/fdb/preprocessing.py

