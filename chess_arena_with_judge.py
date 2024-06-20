import chess
import chess.pgn
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
import chess.pgn
import re
import os
_ = load_dotenv(find_dotenv())

# llm = ChatGoogleGenerativeAI(model="gemini-pro")

white_player = "GPT-4o"
black_player = "Gemini-Pro"
llm1 = ChatOpenAI(temperature=0.1, model='gpt-4o')
llm2 = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-1.5-pro-latest")

# white_player = "Gemini-Pro"
# black_player = "GPT-4o"
# llm1 = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-1.5-pro-latest")
# llm2 = ChatOpenAI(temperature=0.1, model='gpt-4o')

memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
memory2 = ConversationBufferMemory(memory_key="chat_history", input_key="input")

# Definindo os prompts para as LLMs
system_template = """
    You are a Chess Grandmaster.
    We are currently playing chess. 
    You are playing with the {color} pieces.
    
    I will give you the last move, the history of the game so far, the
    actual board position and you must analyze the position and find the best move.

    # OUTPUT
    Do not use any special characters. 
    Give your response in the following order:

    1. Your move, using the following format: My move: "Move" (in the SAN notation, in english).
    2. The explanation, in Portuguese, of why you chose the move, in no more than 3 sentences.
    """

prompt_template1 = ChatPromptTemplate.from_messages([
    ("system", system_template.format(color="white")), 
    ("human", "{input}")])
prompt_template2 = ChatPromptTemplate.from_messages([
    ("system", system_template.format(color="black")), 
    ("human", "{input}")])

# Criando os LLMChains
# chain1 = LLMChain(llm=llm1, prompt=prompt_template1, memory=memory)
# chain2 = LLMChain(llm=llm2, prompt=prompt_template2, memory=memory2)
chain1 = prompt_template1 | llm1
chain2 = prompt_template2 | llm2

judge_template = """
    You are a professional chess arbiter, working on a LLM's Chess Competition.

    Your job is to parse last player's move and ensure that all chess moves are valid and correctly formatted in 
    Standard Algebraic Notation (SAN) for processing by the python-chess library.

    ### Input:
    - Last player's  move
    - List of valid moves in SAN

    ### Output:
    - Return the corresponding move in the list of valid SAN moves.
    - If the proposed move is not in the valid moves list, must respond with "None"

    ### Your turn:
    - Proposed move: {proposed_move}
    - List of valid moves: {valid_moves}

    You should only respond the valid move, without the move number, nothing more.
    Your response:
    """

llm3 = ChatGroq(temperature=0, model_name="llama3-70b-8192")
judge_prompt = PromptTemplate.from_template(template=judge_template)
chain3 = judge_prompt | llm3

move_raw = ""
def get_move(llm_chain, last_move, board, node, color, alert_msg=False):
    global chain3, move_raw
    game_temp = chess.pgn.Game.from_board(board)
    str_board = str(board)
    history = str(game_temp)
    pattern = r".*?(?=1\. e4)"
    history = re.sub(pattern, "", history, flags=re.DOTALL)

    legal_moves = list(board.legal_moves)
    san_moves = str([board.san(move) for move in legal_moves])

    template_input=""" 
        Here's the history of the game:
        {history}

        The last move played was: 
        {last_move}   

        Find the best move.
    """
    
    if not alert_msg:
        user_input = template_input.format(
                                    last_move=last_move,
                                    history=history)
    else:  
        user_input="""
        Here's the actual board position: 
        {str_board}

        Here is the game history so far:
        {history}

        The last move played was: 
        {last_move}   

        Here's a list of valid moves in this position:
        {san_moves}

        You must choose one of the valid moves.
        """.format(san_moves=san_moves, 
                history=history, 
                str_board=str_board,
                last_move=last_move,
                )
        
    
    response = llm_chain.invoke({"input": user_input})
    # move_raw = response["text"].strip()
    move_raw = response.content.strip()
    
    try:
        if alert_msg:
            print("Alerting player!")
            print(move_raw)

        move = chain3.invoke({"proposed_move": move_raw,
                "valid_moves": san_moves
            }).content.strip()
        print(f"Old move: {move_raw}")
        print("-----")
        print(f"New move: {move}")
        
        move_board = board.push_san(move)
        next_node = node.add_variation(move_board)
        next_node.comment = move_raw         
        return move, next_node
    
    except ValueError:
        print(f"Invalid move generated by {color}: {move}")
        return None, node

# Inicializando o tabuleiro de xadrez
for move1 in ["1. e4", "1. d4", "1. c4", "1. Nf3", "1. b3", "1. c3", "1. e3", "1. d3", "1. g3", "1. Nc3"]:
    folder_name = f"{white_player} vs {black_player}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    game_num = max([int(i.split("_")[0]) for i in ["0_0"]+ os.listdir(folder_name)]) + 1

    print("============")
    print(f"New Game with {move1}")
    board = chess.Board()
    
    # Definir o nó atual como o nó raiz do jogo
    game = chess.pgn.Game()
    node = game

    # Loop de jogo
    move_board = board.push_san(move1.split()[-1])
    node = node.add_variation(move_board)

    while not board.is_game_over():    
        move2 = None
        c = 0
        while move2 is None:
            alert = False if c == 0 else True
            move2, node = get_move(chain2, move1, board, node, "black", alert)
            c += 1
        print("\n========================")
        if board.is_game_over():
            break
      
        move1 = None
        c = 0
        while move1 is None:
            alert = False if c == 0 else True
            move1, node = get_move(chain1, move2, board, node, "white", alert)
            c+=1
        print("\n========================")

        if board.is_game_over():
            break
        # game = chess.pgn.Game.from_board(board)
        # print(str(game))
        # print("\n========================")

        # game = chess.pgn.Game.from_board(board)
        game.headers["White"] = white_player
        game.headers["Black"] = black_player
        with open(f"{folder_name}/{game_num}_game.pgn", "w") as f:
            f.write(str(game))
        # print("\n========================")
        # print(game)


    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        result = "1/2-1/2"
    elif board.result() == "1-0":
        result = "1-0"
    else:
        result = "0-1"  
    game.headers["Result"] = result

    with open(f"{folder_name}/{game_num}_game.pgn", "w") as f:
        f.write(str(game))
        

    print("Game Over")
    print(board.result())
