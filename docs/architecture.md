# Arquitetura Reestruturada

## Visao geral

O notebook anexo mistura tres responsabilidades no mesmo fluxo:

- bootstrap tabular de dados financeiros;
- preprocessamento multi-origem de Wi-Fi CSI;
- campanhas experimentais Tx-only vs Tx+CSI.

A reestruturacao separa essas responsabilidades em pacotes explicitos.

## Mapeamento notebook -> codigo

- Celulas 1-12: `cross_layer_csi.tabular` e `cross_layer_csi.pipelines.tabular`
- Celulas 13-26: `cross_layer_csi.csi` e `cross_layer_csi.pipelines.csi`
- Celulas 27-53: `cross_layer_csi.experiments.registry`

## Pastas principais

- `src/cross_layer_csi/core`
  Centraliza `ProjectPaths` e os caminhos usados pelas duas camadas do projeto.

- `src/cross_layer_csi/tabular`
  Mantem o bootstrap transacional atual, agora sob um namespace coerente com o estudo cross-layer.

- `src/cross_layer_csi/csi`
  Contem as classes extraidas do notebook para conversao de amplitudes, filtragem de subportadoras, suavizacao temporal e harmonizacao de granularidade espectral.

- `src/cross_layer_csi/pipelines`
  Reune os fluxos completos de execucao para tabular e CSI.

- `src/cross_layer_csi/experiments`
  Registra as campanhas experimentais existentes no notebook, preparando o terreno para extracao posterior do codigo analitico.

## Convencoes de dados

- `data/raw/` e `data/processed/` continuam dedicados ao fluxo tabular.
- `data/csi/` passa a concentrar fontes CSI, artefatos convertidos e saidas harmonizadas.
- `reports/generated/` agrega tanto os relatórios tabulares quanto o resumo `csi_harmonization_summary.csv`.
