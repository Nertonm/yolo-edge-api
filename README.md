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
