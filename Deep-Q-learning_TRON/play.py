import pygame
from tron.window import Window
from Net.ACNet import Net2
from tron.util import *
from ACKTR import Brain
from Net.DQNNet import Net as DQNNET
import argparse

import random

folderName = 'save'


def random_position(width, height):
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)

    return [x, y]


def display_game_menu(window, game):
    window.screen.fill([0, 0, 0])

    myimage = pygame.image.load("asset/TronTitle.png")
    myimage = pygame.transform.scale(myimage, pygame.display.get_surface().get_size())
    imagerect = myimage.get_rect(center=window.screen.get_rect().center)
    window.screen.blit(myimage, imagerect)

    pygame.display.flip()

    event = pygame.event.poll()
    while 1:
        event = pygame.event.poll()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                window = Window(game, 40)
                break



def print_game_results(game):
    if game.winner is None:
        print("It's a draw!")
    else:
        print('Player {} wins! Duration: {}'.format(game.winner, len(game.history)))


def main(args):
    pygame.init()
    rating=False
    iter=30

    actor_critic = Net2() 
    global_brain = Brain(actor_critic,args, acktr=True)
    # global_brain.actor_critic.load_state_dict(torch.load(folderName + '/ACKTR_player_test.bak'))
    # global_brain.actor_critic.eval()

    actor_critic2 = Net2()  
    global_brain2 = Brain(actor_critic2,args, acktr=True)
    # global_brain2.actor_critic.load_state_dict(torch.load(folderName + '/ACKTR_player_test.bak'))
    # global_brain2.actor_critic.eval()

    # DQN=DQNNET()
    # DQN.load_state_dict(torch.load(folderName+'/DDQN.bak'))
    # DQN.eval()
        event = pygame.event.poll()
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_q:
            pygame.quit()
            quit()

    if rating:
        nullgame = 0
        p1_win = 0
        p2_win = 0

        for i in range(iter):
            game = make_game(False, False, "fair")
            pygame.mouse.set_visible(False)
            window = None

            game.main_loop(global_brain.actor_critic, pop_up, window, DQN, ("AC", "DQN"))

            if game.winner is None:
                nullgame+=1
            elif game.winner ==1:
                p1_win+=1
            else:
                p2_win+=1

        print("Player 1:{} \n Player 2:{}\n ".format(p1_win,p2_win))
    else:
        while True:
            game = make_game(False, False, "fair")
            pygame.mouse.set_visible(False)

            window = Window(game, 40)

            game.main_loop(global_brain.actor_critic,pop_up,window,global_brain2.actor_critic)
            print_game_results(game)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', required=False, help='model structure number')
    parser.add_argument('-r', required=False, help='reward condition number')

    parser.add_argument('-p', required=False, help='policy coefficient')
    parser.add_argument('-v', required=False, help='value coefficient')
    parser.add_argument('-u', required=False, help='unique string')

    args = parser.parse_args()


    main(args)
