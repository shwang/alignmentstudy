from pacman import readCommand, runGames

def interactive():
    args = readCommand("--layout testMaze --pacman KeyboardAgent --createPolicy".split())
    return runGames( **args )

if __name__ == '__main__':
    interactive()
