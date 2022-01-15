# Named-entity recognition

## Description
Данная модель осуществляет в тексте поиск именнованных сущностей следующего вида:
  1. даты и все, что к ним относится (например, на следующей неделе - дата)
  2. именованные личности (например, Робин Гуд - именованная личность)
  3. наименования организаций

В данной работе применяется архитектура BiLSTM + CRF, а также используется BERT-embedding:
![alt text](image/picture.png)
## Install and launch

    git clone https://github.com/prokopk1n/bilstm
    python ./main.py -i input.txt -o output_file.txt
