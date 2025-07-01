# PROCESSO ESCALÁVEL PARA OBTENÇÃO DE PERFIS ALTIMÉTRICOS UTILIZANDO MODELOS DIGITAIS DE ELEVAÇÃO OPEN SOURCE

> Este repositório contém o código relacionado ao trabalho enviado para a [Anpet 2025](https://www.eventweb.com.br/anpet2025/home-event/)

## Visão Geral

Um processo escalável para calcular perfis altimétricos e localizar, sobre esses perfis, as paradas das linhas de ônibus, utilizando Modelos Digitais de Elevação (MDE). 

## Como utilizar

### Criar chave de API no OpenTopography

- Acesse: https://portal.opentopography.org/.
- Crie uma conta (se ainda não tiver).
- Após o login, vá em "My Account" e gere uma nova API key.
- Crie um arquivo `.env` baseado no arquivo `.env.example` na raiz do projeto.
- Preencha a variável de ambiente `API_KEY_OPEN_TOPOGRAPHY` com a chave gerada.

### Configurações do projeto

#### Pré-requisitos

- [Python +3.13](https://www.python.org/)
- [Poetry](https://python-poetry.org/)

#### Instalar dependências

- A partir de um terminal na raiz do projeto, execute o comando `poetry shell` para ativar o ambiente local
- Posteriormente, execute o comando `poetry install` para instalar as dependências necessárias

### Preparação do Arquivo KML (Construção do Traçado da Linha de Ônibus)

Utilize o Google Earth para desenhar manualmente o percurso da linha.
O traçado deve ser feito com precisão, marcando:

- Paradas: locais de embarque/desembarque.
- Pontos de Referência: vértices onde há mudança de direção.

Cada ponto deve ser nomeado no seguinte formato: `[tipo do ponto] [número do tipo]`:

- Tipo do ponto: "Parada" ou "Ref".
- Número do tipo: contador dentro da categoria.

Exemplos válidos:

- Parada 1
- Ref 1
- Ref 2
- Parada 2

> A sequência deve seguir a ordem gps do percurso. 

Exporte e salve o traçado no formato KML na pasta: assets/dados_entrada

> Se houver um arquivo com dados de rotas de GPS para fins de comparação, o arquivo deve conter ao menos a seguinte organização dos pontos [número sequencial] [tipo do ponto]

#### Executando o projeto

- A partir de um terminal na raiz do projeto, execute o comando `poetry shell` para ativar o ambiente local
- Posteriormente, execute o comando `python main.py` para executar os scripts

## Licença

Este projeto é licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para maiores detalhes.

## Contato

Para qualquer dúvida ou problema, por favor, abra uma <i>issue</i> neste repositório ou contate os mantenedores.
