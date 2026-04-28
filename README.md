# Cross-Layer CSI Anomaly Detection

Projeto reestruturado a partir do notebook de referencia **Cross-Layer Physical Presence Attestation via Wi-Fi CSI for In-Person Payment Anomaly Detection**. A base agora organiza explicitamente as duas camadas do estudo:

- `tabular`: bootstrap e preparo dos datasets transacionais.
- `csi`: aquisicao, conversao, filtragem, suavizacao e harmonizacao dos traces de Wi-Fi CSI.

## O que mudou

- O ponto de entrada publico passou a ser `src/cross_layer_csi/`.
- O bootstrap tabular anterior foi preservado e encapsulado como a camada `cross_layer_csi.tabular`.
- As classes centrais de preprocessamento CSI foram extraidas do notebook para modulos Python reutilizaveis.
- As etapas experimentais do notebook foram mapeadas em `cross_layer_csi.experiments`.

## Estrutura

- `src/cross_layer_csi/core/`: caminhos e infraestrutura comum do projeto.
- `src/cross_layer_csi/tabular/`: pipeline transacional e adaptadores para os datasets IEEE-CIS, Sparkov e E-commerce.
- `src/cross_layer_csi/csi/`: download, analise e preprocessamento CSI.
- `src/cross_layer_csi/pipelines/`: orquestracao de alto nivel para tabular e CSI.
- `src/cross_layer_csi/experiments/`: catalogo das etapas experimentais derivadas do notebook.
- `data/raw/`: dados tabulares brutos.
- `data/processed/`: artefatos tabulares preparados.
- `data/csi/`: materias-primas e artefatos intermediarios de CSI.
- `reports/generated/`: sumarios tabulares e relatorios do preprocessamento CSI.
- `docs/architecture.md`: mapa entre o notebook anexo e a estrutura atual do pacote.

## Como executar

```powershell
python -m pip install -e .[dev]
python -m cross_layer_csi.cli tabular-bootstrap --datasets all
python -m cross_layer_csi.cli csi-preprocess --no-plots
```

## Compatibilidade

O pacote antigo `multidataset_fraud` foi mantido no repositório para preservar a logica tabular ja validada durante a migracao. O namespace recomendado para novos usos e `cross_layer_csi`.
