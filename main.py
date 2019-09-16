import numpy as np
import pyfiglet


# ⡿⠋⠄⣀⣀⣤⣴⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣌⠻⣿⣿
# ⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠹⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠹
# ⣿⣿⡟⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡛⢿⣿⣿⣿⣮⠛⣿⣿⣿⣿⣿⣿⡆
# ⡟⢻⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣣⠄⡀⢬⣭⣻⣷⡌⢿⣿⣿⣿⣿⣿
# ⠃⣸⡀⠈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠈⣆⢹⣿⣿⣿⡈⢿⣿⣿⣿⣿
# ⠄⢻⡇⠄⢛⣛⣻⣿⣿⣿⣿⣿⣿⣿⣿⡆⠹⣿⣆⠸⣆⠙⠛⠛⠃⠘⣿⣿⣿⣿
# ⠄⠸⣡⠄⡈⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⠁⣠⣉⣤⣴⣿⣿⠿⠿⠿⡇⢸⣿⣿⣿
# ⠄⡄⢿⣆⠰⡘⢿⣿⠿⢛⣉⣥⣴⣶⣿⣿⣿⣿⣻⠟⣉⣤⣶⣶⣾⣿⡄⣿⡿⢸
# ⠄⢰⠸⣿⠄⢳⣠⣤⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⣼⣿⣿⣿⣿⣿⣿⡇⢻⡇⢸
# ⢷⡈⢣⣡⣶⠿⠟⠛⠓⣚⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⢸⠇⠘
# ⡀⣌⠄⠻⣧⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠛⠛⠛⢿⣿⣿⣿⣿⣿⡟⠘⠄⠄
# ⣷⡘⣷⡀⠘⣿⣿⣿⣿⣿⣿⣿⣿⡋⢀⣠⣤⣶⣶⣾⡆⣿⣿⣿⠟⠁⠄⠄⠄⠄
# ⣿⣷⡘⣿⡀⢻⣿⣿⣿⣿⣿⣿⣿⣧⠸⣿⣿⣿⣿⣿⣷⡿⠟⠉⠄⠄⠄⠄⡄⢀
# ⣿⣿⣷⡈⢷⡀⠙⠛⠻⠿⠿⠿⠿⠿⠷⠾⠿⠟⣛⣋⣥⣶⣄⠄⢀⣄⠹⣦⢹⣿

# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣬⡛⣿⣿⣿⣯⢻ 
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢻⣿⣿⢟⣻⣿⣿⣿⣿⣿⣿⣮⡻⣿⣿⣧ 
# ⣿⣿⣿⣿⣿⢻⣿⣿⣿⣿⣿⣿⣆⠻⡫⣢⠿⣿⣿⣿⣿⣿⣿⣿⣷⣜⢻⣿ 
# ⣿⣿⡏⣿⣿⣨⣝⠿⣿⣿⣿⣿⣿⢕⠸⣛⣩⣥⣄⣩⢝⣛⡿⠿⣿⣿⣆⢝ 
# ⣿⣿⢡⣸⣿⣏⣿⣿⣶⣯⣙⠫⢺⣿⣷⡈⣿⣿⣿⣿⡿⠿⢿⣟⣒⣋⣙⠊ 
# ⣿⡏⡿⣛⣍⢿⣮⣿⣿⣿⣿⣿⣿⣿⣶⣶⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿ 
# ⣿⢱⣾⣿⣿⣿⣝⡮⡻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⣋⣻⣿⣿⣿⣿ 
# ⢿⢸⣿⣿⣿⣿⣿⣿⣷⣽⣿⣿⣿⣿⣿⣿⣿⡕⣡⣴⣶⣿⣿⣿⡟⣿⣿⣿ 
# ⣦⡸⣿⣿⣿⣿⣿⣿⡛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣿⣿⣿ 
# ⢛⠷⡹⣿⠋⣉⣠⣤⣶⣶⣿⣿⣿⣿⣿⣿⡿⠿⢿⣿⣿⣿⣿⣿⣷⢹⣿⣿ 
# ⣷⡝⣿⡞⣿⣿⣿⣿⣿⣿⣿⣿⡟⠋⠁⣠⣤⣤⣦⣽⣿⣿⣿⡿⠋⠘⣿⣿ 
# ⣿⣿⡹⣿⡼⣿⣿⣿⣿⣿⣿⣿⣧⡰⣿⣿⣿⣿⣿⣹⡿⠟⠉⡀⠄⠄⢿⣿ 
# ⣿⣿⣿⣽⣿⣼⣛⠿⠿⣿⣿⣿⣿⣿⣯⣿⠿⢟⣻⡽⢚⣤⡞⠄⠄⠄⢸⣿

class Board():

    def __init__(self, n, plname):
        self.dimension = n
        self.players = []
        for y in plname:
            pl = Player(y)
            self.players.append(pl)
        self.idxofcurrentplayer = 0
        self.currentplayer = self.players[self.idxofcurrentplayer]
        self.emptytile = "x"
        self.show(n)

    def show(self, n):
        self.boardmatrix = []
        self.spiece = []
        self.opiece = []
        
        for x in range(n):
            self.boardmatrix.append([])
            for _ in range(n):
                self.boardmatrix[x].append(self.emptytile)
        
    def printClass(self):
        for x in self.boardmatrix:
            for y in x:
                print(y, end = ' ')
            print()
    
    def checkadd(self, x, y):
        if self.boardmatrix[y][x] != self.emptytile:
            return False
        else:
            return True

    def adds(self, coord):
        x = int(coord.split(' ')[0]) - 1
        y = int(coord.split(' ')[1]) - 1
        if not self.checkadd(x, y):
            print("Look you pepeg \n slot has been taken")
            return
        self.boardmatrix[y][x] = 'S'
        self.checkscores(y, x)
        self.nextplayer()

    def addo(self, coord):
        x = int(coord.split(' ')[0]) - 1
        y = int(coord.split(' ')[1]) - 1
        if not self.checkadd(x, y):
            print("Look you pepeg \n slot has been taken")
            return
        self.boardmatrix[y][x] = 'O'
        self.checkscoreo(y, x)
        self.nextplayer()

    def checkscores(self, y, x):
        for modify in range(-1, 2):
            for modifx in range(-1, 2):
                if modifx != 0 or modify != 0:
                    try:
                        if self.boardmatrix[y + modify][x + modifx] == 'O' and  self.boardmatrix[y + modify*2][x + modifx*2] == 'S':
                            self.currentplayer.addscore(1)
                    except:
                        pass
    
    def checkscoreo(self, y, x):
        for modify in range(-1, 2):
            for modifx in range(-1, 2):
                if modifx != 0 or modify != 0:
                    try:
                        if self.boardmatrix[y + modify][x + modifx] == 'S' and  self.boardmatrix[y - modify][x - modifx] == 'S':
                            self.currentplayer.addscore(1)
                    except:
                        pass 
                    
    def nextplayer(self):
        self.idxofcurrentplayer = (self.idxofcurrentplayer + 1) % len(self.players)
        self.currentplayer = self.players[self.idxofcurrentplayer]

    def printscore(self):
        print('Current Score is :')
        i = 0
        for x in self.players:
            i = i + 1
            print('Player ' + str(i) + " " + x.name + ' Score : ' + str(x.myscore))


    def checkinput(self, coord):
        try:        
            a = int(coord.split(' ')[0])
            b = int(coord.split(' ')[1])
            _ = self.boardmatrix[b][a]
            return True
        except:
            print("Input must be 2 integer coordinate between 1 and " + str(self.dimension))
            print()
            return False



            
        

class Player():

    def __init__(self, name):
        self.name = name
        self.myscore = 0 

    def addscore(self, n):
        self.myscore = self.myscore + n

def main():
    ascii_banner = pyfiglet.figlet_format("SOS GEEMU!")
    print(ascii_banner)

    plname = []
    print("You need at least 1 player")
    inval = ""
    loopi = 0
    while 1:
        loopi = loopi + 1
        inval = input("Enter name for player " + str(loopi) + " (input 'stop' to stop player input)")
        if inval == "stop":
            break
        plname.append(inval)
    sosboard = Board(10, plname)
    while 1:
        print("Your turn:" + sosboard.currentplayer.name)
        val = input("Add 'S' or 'O'\n") 
        if val in ['S', 's']:
            coord = input("Set coordinate, (x y):")
            if sosboard.checkinput(coord):
                sosboard.adds(coord)
                sosboard.printClass()
                sosboard.printscore()

        elif val in ['O', 'o']:
            coord = input("Set coordinate, (x y):")
            if sosboard.checkinput(coord):
                sosboard.addo(coord)
                sosboard.printClass()
                sosboard.printscore()
        else:
            print('S or O, cmonBruh')
        
        print("\n\n")
        
    print('done')

main()