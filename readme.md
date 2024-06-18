# LLM Chess Arena

<img src="./images/video.gif"/>

Este é o código fonte do projeto apresentado neste vídeo:
<br>
https://www.instagram.com/reel/C8Ndmh2OAze/

Este é um script que permite duas LLMs joguem Xadrez, nos permitindo (de maneira simplificada) comparar dois modelos de linguagem.
Adicionei um histórico de 20 partidas jogadas o ChatGPT-4o e Gemini-1.5 Pro.



## Como Rodar?

1. Clone o projeto.
2. Crie chaves de acesso para o ChatGPT e o Gemini.
3. Na pasta do projeto crie um arquivo chamado .env
4. Coloque suas API Key, no seguinte formato:

```
GOOGLE_API_KEY=sua-chave
OPENAI_API_KEY=sua-chave
```

5. Instale as dependências, abrindo seu terminal e usando o comando:

`pip install -r requirements.txt`

6. Execute o script `chess_arena_with_judge.py`

