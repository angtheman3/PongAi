import pygame
import random

# Constants
FPS = 60
WIN_WIDTH, WIN_HEIGHT = 400, 400
PADDLE_W, PADDLE_H = 10, 60
PADDLE_OFFSET = 10
BALL_W, BALL_H = 10, 10
PADDLE_SPEED = 2
BALL_SPEED_X, BALL_SPEED_Y = 3, 2
WHITE, BLACK = (255, 255, 255), (0, 0, 0)

# Initialize screen
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))


def draw_rect(x, y, width, height, color):
    pygame.draw.rect(screen, color, pygame.Rect(x, y, width, height))


def update_ball(p1_y, p2_y, ball_x, ball_y, dir_x, dir_y):
    ball_x += dir_x * BALL_SPEED_X
    ball_y += dir_y * BALL_SPEED_Y
    score = 0

    if PADDLE_OFFSET + PADDLE_W >= ball_x and p1_y <= ball_y <= p1_y + PADDLE_H:
        dir_x = 1
    elif ball_x <= 0:
        dir_x = 1
        score = -1
    elif WIN_WIDTH - PADDLE_W - PADDLE_OFFSET <= ball_x and p2_y <= ball_y <= p2_y + PADDLE_H:
        dir_x = -1
    elif ball_x >= WIN_WIDTH - BALL_W:
        dir_x = -1
        score = 1
    if ball_y <= 0 or ball_y >= WIN_HEIGHT - BALL_H:
        dir_y *= -1

    return score, ball_x, ball_y, dir_x, dir_y


def update_paddle(y, action, up_act, down_act):
    if action[up_act] == 1:
        y -= PADDLE_SPEED
    if action[down_act] == 1:
        y += PADDLE_SPEED
    return max(0, min(WIN_HEIGHT - PADDLE_H, y))


class Pong:
    def __init__(self):
        self.score = 0
        self.p1_y, self.p2_y = (WIN_HEIGHT - PADDLE_H) // 2, (WIN_HEIGHT - PADDLE_H) // 2
        self.ball_x, self.ball_y = WIN_WIDTH // 2, random.randint(0, WIN_HEIGHT - BALL_H)
        self.dir_x, self.dir_y = random.choice([-1, 1]), random.choice([-1, 1])

    def get_frame(self):
        pygame.event.pump()
        screen.fill(BLACK)
        draw_rect(PADDLE_OFFSET, self.p1_y, PADDLE_W, PADDLE_H, WHITE)
        draw_rect(WIN_WIDTH - PADDLE_OFFSET - PADDLE_W, self.p2_y, PADDLE_W, PADDLE_H, WHITE)
        draw_rect(self.ball_x, self.ball_y, BALL_W, BALL_H, WHITE)
        pygame.display.flip()
        return pygame.surfarray.array3d(pygame.display.get_surface())

    def get_next_frame(self, action):
        pygame.event.pump()
        screen.fill(BLACK)
        self.p1_y = update_paddle(self.p1_y, action, 1, 2)
        self.p2_y = update_paddle(self.p2_y, action, 3, 4)
        score, self.ball_x, self.ball_y, self.dir_x, self.dir_y = update_ball(self.p1_y, self.p2_y, self.ball_x, self.ball_y, self.dir_x, self.dir_y)
        self.score += score
        print(f"Score: {self.score}")
        draw_rect(self.ball_x, self.ball_y, BALL_W, BALL_H, WHITE)
        pygame.display.flip()
        return score, pygame.surfarray.array3d(pygame.display.get_surface())


if __name__ == "__main__":
    game = Pong()
    game.get_frame()

