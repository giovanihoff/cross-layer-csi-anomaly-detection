from __future__ import annotations

from .models import DatasetResearch


RESEARCH_NOTES: dict[str, DatasetResearch] = {
    "ieee_cis": DatasetResearch(
        dataset_key="ieee_cis",
        display_name="IEEE-CIS Fraud Detection",
        uid_choice="uid = card1_addr1 + floor(TransactionDT/86400 - D1)",
        uid_rationale=(
            "O consenso mais forte encontrado na comunidade Kaggle para o IEEE-CIS e usar um UID "
            "proxy baseado em identidade de cartao/endereco combinado com a ancora temporal derivada de D1. "
            "Essa formulacao aparece nas discussoes dos vencedores e em resumos tecnicos que espelham a solucao "
            "vencedora. O projeto tambem preserva `TransactionID` como `event_id` nativo para separar identidade "
            "da transacao e identidade do usuario."
        ),
        training_prep=(
            "A preparacao inicial replica o nucleo do pipeline vencedor amplamente citado: merge de `train_transaction` "
            "com `train_identity`, construcao do UID, normalizacao dos campos `D*` por dia relativo, derivacao de "
            "features temporais de `TransactionDT` e agregacoes por UID e `card1` para `TransactionAmt`."
        ),
        references=(
            {
                "label": "Kaggle Competition Overview",
                "url": "https://www.kaggle.com/competitions/ieee-fraud-detection/overview",
            },
            {
                "label": "Kaggle Kernel - xgb-fraud-with-magic-0-9600",
                "url": "https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600",
            },
            {
                "label": "Kaggle Discussion - 1st Place Solution Part 1",
                "url": "https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284",
            },
            {
                "label": "Kaggle Discussion - 1st Place Solution Part 2",
                "url": "https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308",
            },
            {
                "label": "Mirror summarizing winner UID strategy",
                "url": "https://github.com/zehuichen123/blog/blob/67d8286f9ca254e4a60d572b708030d4496af9d7/_posts/2019-10-13-relearning-tabular-data-competition-kaggle-ieee-fraud-detection.md",
            },
            {
                "label": "Fraud Dataset Benchmark IEEE preprocessing",
                "url": "https://github.com/amazon-science/fraud-dataset-benchmark/blob/main/src/fdb/preprocessing.py",
            },
        ),
    ),
    "sparkov": DatasetResearch(
        dataset_key="sparkov",
        display_name="Sparkov Simulated Credit Card Transactions",
        uid_choice="uid = cc_num; event_id = trans_num",
        uid_rationale=(
            "Nao existe uma competicao Kaggle oficial para essa base com solucao vencedora canonica. Para evitar UID "
            "artificial, a estrategia adotada usa os identificadores nativos do proprio dataset: `trans_num` como "
            "identificador unico da transacao e `cc_num` como identidade recorrente do cartao/cliente. `merchant` "
            "permanece como entidade secundaria para agregacoes de risco."
        ),
        training_prep=(
            "O preparo preserva o split original `fraudTrain.csv`/`fraudTest.csv`, converte timestamps e data de "
            "nascimento, deriva idade no momento da transacao, cria distancia geografica usuario-merchant e produz "
            "agregacoes por `uid` e `merchant` sobre `amt`."
        ),
        references=(
            {
                "label": "Kaggle Dataset - kartik2112/fraud-detection",
                "url": "https://www.kaggle.com/datasets/kartik2112/fraud-detection",
            },
            {
                "label": "Fraud Dataset Benchmark README",
                "url": "https://github.com/amazon-science/fraud-dataset-benchmark",
            },
            {
                "label": "Fraud Dataset Benchmark preprocessing",
                "url": "https://github.com/amazon-science/fraud-dataset-benchmark/blob/main/src/fdb/preprocessing.py",
            },
        ),
    ),
    "ecommerce": DatasetResearch(
        dataset_key="ecommerce",
        display_name="Fraud E-commerce",
        uid_choice="uid = device_id; account_id = user_id; event_id = deterministic row-level hash/index",
        uid_rationale=(
            "Tambem nao ha competicao Kaggle oficial com vencedor canonico. Para esta base, o projeto evita sintetizar "
            "um usuario inexistente e separa os papeis: `device_id` vira o UID operacional principal por ser a entidade "
            "mais reutilizavel em fraude online, `user_id` e mantido como identidade de conta, e o `event_id` e criado "
            "de forma deterministica porque a fonte nao traz um transaction id nativo explicito."
        ),
        training_prep=(
            "O preparo replica o conjunto de features mais recorrente nessa base: parsing temporal, `time_since_signup`, "
            "contagens por dispositivo/IP/usuario e, quando o arquivo auxiliar estiver presente, enriquecimento por pais "
            "via faixas de IP."
        ),
        references=(
            {
                "label": "Kaggle Dataset - vbinh002/fraud-ecommerce",
                "url": "https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce",
            },
            {
                "label": "Fraud Dataset Benchmark preprocessing",
                "url": "https://github.com/amazon-science/fraud-dataset-benchmark/blob/main/src/fdb/preprocessing.py",
            },
        ),
    ),
}


def get_research(dataset_key: str) -> DatasetResearch:
    return RESEARCH_NOTES[dataset_key]

