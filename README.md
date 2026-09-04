# yolo-edge-api

## Preprocessing do dataset

O dataset de segurança em construções foi preparado no Roboflow antes da exportação para YOLOv8.

Configurações aplicadas:

- Auto-Orient.
- Resize para `640x480` com `Stretch`.
- Auto-Adjust Contrast com `Adaptive Equalization`.
- Flip horizontal.
- Rotação entre `-15°` e `+15°`.
- Shear horizontal e vertical de `±10°`.
- Saturação entre `-25%` e `+25%`.
- Brilho entre `-25%` e `+25%`.
- Blur de até `1.5px`.
- Noise de até `1.49%` dos pixels.
- Três variações por imagem de treino.

O split original foi de 70% para treino, 15% para validação e 15% para teste: 844, 181 e 181 imagens. Como o augmentation foi aplicado somente ao treino, a versão exportada ficou com 2532 imagens de treino, 181 de validação e 181 de teste, totalizando 2894 imagens.

O projeto original possuía cinco classes. Após a exportação, as labels foram filtradas e remapeadas para as três classes usadas no projeto: `Capacete`, `Colete` e `Pessoa`. As classes `no-helmet` e `no-vest` foram removidas, sem deixar imagens sem label.

A estrutura final foi validada pelo `scripts/inspect_dataset.py`: 2894 imagens, 0 imagens sem label e dataset aprovado para treinamento.

## Dataset usado no treinamento

Arquivo: `dataset/exports/epi-v1` (versionado via DVC, remote `local_remote`).

Origem: Roboflow Universe, projeto `construction-safety-gsnvb-d5kkj`
(workspace `thiago-nerton-macedo-alves`, versao 1, licenca CC BY 4.0).
URL: https://universe.roboflow.com/thiago-nerton-macedo-alves/construction-safety-gsnvb-d5kkj/dataset/1

Propriedades da versao exportada:

- 2894 imagens: 2532 treino, 181 validacao, 181 teste.
- Resolucao uniforme 640x480.
- Tres classes: `Capacete`, `Colete`, `Pessoa` (arquivo `data.yaml`, nc=3).
- Uma anotacao YOLO (txt) por imagem; 0 imagens sem label.

Instancias por classe:

| Split | Capacete | Colete | Pessoa |
|-------|----------|--------|--------|
| train | 5352     | 2701   | 5919   |
| valid | 360      | 221    | 398    |
| test  | 399      | 221    | 446    |

Classe minoritaria: `Colete` (19% das instancias de treino).

Verificacoes de integridade:

- 0 duplicatas exatas (md5) entre train, valid e test.
- Near-duplicates (dHash) entre splits: 1 par train/valid e 1 grupo
  train/test, todos frames de sequencia da mesma cena (`ppe_NNNN`).
  Impacto menor que 0,1% do treino.
- Variantes repetidas dentro de train sao saida da augmentation do Roboflow
  (3 variacoes por imagem), sem cruzamento de splits.

Metricas do modelo treinado com este dataset estao em
`runs/detect/runs/epi-v1/` (val mAP@0.5 0.934; test mAP@0.5 0.886).
