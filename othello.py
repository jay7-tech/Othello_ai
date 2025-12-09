import pygame
import sys
import copy
import time

# --- CONFIGURATION ---
BOARD_SIZE = 8
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 700  # Extra space for score/status
SQUARE_SIZE = WINDOW_WIDTH // BOARD_SIZE

# Colors
GREEN_BOARD = (34, 139, 34)     # Classic Felt Green
GRID_LINE   = (0, 100, 0)       # Darker Green
BLACK       = (0, 0, 0)
WHITE       = (255, 255, 255)
GREY        = (128, 128, 128)   # For valid move hints
TEXT_COLOR  = (255, 255, 255)
BG_COLOR    = (50, 50, 50)      # UI Background

# Game Constants
EMPTY = 0
BLACK_PLAYER = 1  # Human
WHITE_PLAYER = 2  # AI

# --- LOGIC CLASS ---
class OthelloGame:
    def __init__(self):
        self.board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.turn = BLACK_PLAYER
        self.history = []
        self._init_board()

    def _init_board(self):
        # Othello starts with 4 pieces in the center
        mid = BOARD_SIZE // 2
        self.board[mid-1][mid-1] = WHITE_PLAYER
        self.board[mid][mid] = WHITE_PLAYER
        self.board[mid-1][mid] = BLACK_PLAYER
        self.board[mid][mid-1] = BLACK_PLAYER

    def get_valid_moves(self, player):
        """
        Returns a list of (r, c) where the player can legally place a piece.
        Rule: Must flank at least one opponent piece.
        """
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.is_valid_move(r, c, player):
                    moves.append((r, c))
        return moves

    def is_valid_move(self, r, c, player):
        if self.board[r][c] != EMPTY:
            return False
        
        opponent = 2 if player == 1 else 1
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            found_opponent = False
            
            # Walk in direction
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if self.board[nr][nc] == opponent:
                    found_opponent = True
                elif self.board[nr][nc] == player:
                    if found_opponent:
                        return True # Found sandwich
                    else:
                        break # Found my own piece immediately
                else:
                    break # Found empty space
                nr += dr
                nc += dc
        return False

    def make_move(self, r, c, player):
        self.board[r][c] = player
        opponent = 2 if player == 1 else 1
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Flip pieces
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            pieces_to_flip = []
            
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if self.board[nr][nc] == opponent:
                    pieces_to_flip.append((nr, nc))
                elif self.board[nr][nc] == player:
                    # Valid sandwich end found, flip all in between
                    for fr, fc in pieces_to_flip:
                        self.board[fr][fc] = player
                    break
                else:
                    break # Hit empty space, invalid direction
                nr += dr
                nc += dc
        
        self.turn = opponent
        return True

    def get_score(self):
        b_count = sum(row.count(BLACK_PLAYER) for row in self.board)
        w_count = sum(row.count(WHITE_PLAYER) for row in self.board)
        return b_count, w_count

    def is_game_over(self):
        # Game over if neither player can move
        black_moves = self.get_valid_moves(BLACK_PLAYER)
        white_moves = self.get_valid_moves(WHITE_PLAYER)
        return not black_moves and not white_moves

# --- AI ENGINE ---
class OthelloAI:
    def __init__(self, player, depth=3):
        self.player = player
        self.opponent = 1 if player == 2 else 2
        self.max_depth = depth
        
        # Positional Weights: Corners are amazing, X-squares (near corners) are terrible
        self.WEIGHTS = [
            [100, -20, 10,  5,  5, 10, -20, 100],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [ 10,  -2, -1, -1, -1, -1,  -2,  10],
            [  5,  -2, -1, -1, -1, -1,  -2,   5],
            [  5,  -2, -1, -1, -1, -1,  -2,   5],
            [ 10,  -2, -1, -1, -1, -1,  -2,  10],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [100, -20, 10,  5,  5, 10, -20, 100]
        ]

    def evaluate(self, game):
        """
        Heuristic:
        1. Material Difference (Coin count)
        2. Positional Strategy (Weighted board)
        """
        b_score, w_score = game.get_score()
        
        # End-game: Prioritize just winning (coin count)
        total_pieces = b_score + w_score
        if total_pieces > 55:
            if self.player == WHITE_PLAYER:
                return w_score - b_score
            else:
                return b_score - w_score

        # Mid-game: Prioritize Position Strategy
        position_score = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == self.player:
                    position_score += self.WEIGHTS[r][c]
                elif game.board[r][c] == self.opponent:
                    position_score -= self.WEIGHTS[r][c]
        
        return position_score

    def minimax(self, game, depth, alpha, beta, maximizing):
        if depth == 0 or game.is_game_over():
            return self.evaluate(game), None

        # Determine current player based on recursion level
        current_player = self.player if maximizing else self.opponent
        valid_moves = game.get_valid_moves(current_player)

        # If no moves, pass turn (recurse without moving)
        if not valid_moves:
            # Check if opponent also has no moves (Game Over check handled at start)
            return self.minimax(game, depth - 1, alpha, beta, not maximizing)

        best_move = None

        if maximizing:
            max_eval = -float('inf')
            for r, c in valid_moves:
                # Create a copy to simulate move
                new_game = copy.deepcopy(game)
                new_game.make_move(r, c, current_player)
                
                eval_score, _ = self.minimax(new_game, depth - 1, alpha, beta, False)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = (r, c)
                
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for r, c in valid_moves:
                new_game = copy.deepcopy(game)
                new_game.make_move(r, c, current_player)
                
                eval_score, _ = self.minimax(new_game, depth - 1, alpha, beta, True)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = (r, c)
                
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return min_eval, best_move

# --- GUI ---
def draw_board(screen, game, valid_moves):
    # Draw Background
    pygame.draw.rect(screen, GREEN_BOARD, (0, 0, WINDOW_WIDTH, WINDOW_WIDTH))
    
    # Draw Grid Lines
    for i in range(BOARD_SIZE + 1):
        # Horizontal
        pygame.draw.line(screen, GRID_LINE, (0, i * SQUARE_SIZE), (WINDOW_WIDTH, i * SQUARE_SIZE), 2)
        # Vertical
        pygame.draw.line(screen, GRID_LINE, (i * SQUARE_SIZE, 0), (i * SQUARE_SIZE, WINDOW_WIDTH), 2)

    # Draw Pieces
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            x = c * SQUARE_SIZE + SQUARE_SIZE // 2
            y = r * SQUARE_SIZE + SQUARE_SIZE // 2
            radius = SQUARE_SIZE // 2 - 5
            
            if game.board[r][c] == BLACK_PLAYER:
                pygame.draw.circle(screen, BLACK, (x, y), radius)
            elif game.board[r][c] == WHITE_PLAYER:
                pygame.draw.circle(screen, WHITE, (x, y), radius)
    
    # Draw Move Hints (Small Grey Dots)
    for r, c in valid_moves:
        x = c * SQUARE_SIZE + SQUARE_SIZE // 2
        y = r * SQUARE_SIZE + SQUARE_SIZE // 2
        pygame.draw.circle(screen, GREY, (x, y), 5)

def draw_status(screen, game, font, status_msg):
    # Bottom UI Panel
    panel_rect = pygame.Rect(0, WINDOW_WIDTH, WINDOW_WIDTH, WINDOW_HEIGHT - WINDOW_WIDTH)
    pygame.draw.rect(screen, BG_COLOR, panel_rect)
    
    b_score, w_score = game.get_score()
    
    score_text = font.render(f"Black: {b_score}  |  White: {w_score}", True, TEXT_COLOR)
    msg_text = font.render(status_msg, True, (255, 255, 0)) # Yellow for status
    
    screen.blit(score_text, (20, WINDOW_WIDTH + 15))
    screen.blit(msg_text, (20, WINDOW_WIDTH + 45))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Othello AI - Minimax Strategy")
    font = pygame.font.SysFont('Arial', 24, bold=True)
    clock = pygame.time.Clock()

    game = OthelloGame()
    ai = OthelloAI(WHITE_PLAYER, depth=3) # Depth 3 is good for Othello
    
    running = True
    game_over = False
    status_msg = "Your Turn (Black)"
    
    while running:
        # Input Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and game.turn == BLACK_PLAYER:
                x, y = pygame.mouse.get_pos()
                if y < WINDOW_WIDTH: # Clicked on board
                    c = x // SQUARE_SIZE
                    r = y // SQUARE_SIZE
                    
                    if game.is_valid_move(r, c, BLACK_PLAYER):
                        game.make_move(r, c, BLACK_PLAYER)
                        if game.is_game_over():
                            game_over = True
                        else:
                            status_msg = "AI Thinking..."
        
        # AI Turn Logic
        if not game_over and game.turn == WHITE_PLAYER:
            # Force draw to show "Thinking"
            valid_moves = game.get_valid_moves(BLACK_PLAYER) # Hint irrelevant for AI turn
            draw_board(screen, game, [])
            draw_status(screen, game, font, status_msg)
            pygame.display.flip()
            
            # Check if AI has moves
            ai_moves = game.get_valid_moves(WHITE_PLAYER)
            if ai_moves:
                # time.sleep(0.5) # Optional delay to make it feel human
                score, move = ai.minimax(game, ai.max_depth, -float('inf'), float('inf'), True)
                if move:
                    game.make_move(move[0], move[1], WHITE_PLAYER)
            else:
                status_msg = "AI has no moves! Passing..."
                game.turn = BLACK_PLAYER
                time.sleep(1) # Read time
            
            if game.is_game_over():
                game_over = True
            elif not game.get_valid_moves(BLACK_PLAYER):
                # If Human has no moves, pass back to AI immediately
                status_msg = "You have no moves! Passing..."
                game.turn = WHITE_PLAYER
            else:
                status_msg = "Your Turn (Black)"

        # Win Condition Text
        if game_over:
            b, w = game.get_score()
            if b > w: res = "YOU WIN!"
            elif w > b: res = "AI WINS!"
            else: res = "DRAW!"
            status_msg = f"GAME OVER: {res}"

        # Rendering
        if not game_over and game.turn == BLACK_PLAYER:
            hints = game.get_valid_moves(BLACK_PLAYER)
        else:
            hints = []
            
        draw_board(screen, game, hints)
        draw_status(screen, game, font, status_msg)
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()