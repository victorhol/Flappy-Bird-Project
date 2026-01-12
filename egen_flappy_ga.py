#Import af pakker
import torch
import torch.nn as nn
import random
from Flappy_bird_Game import Bird, Tunnel


POP_SIZE = 100 #Størrelse på populationen
GENERATIONS = 100 #Antal generationer
MUT_RATE = 0.1 #Mutationsrate
SCALE = 0.2 #Hvor meget mutationerne skal påvirke genomet
MAX_FRAMES = 5000 #Maksimalt antal frames per spil

class FlappyNet(nn.Module):
    def __init__(self):
        super().__init__() #Initialiserer forældren, classen nn.Module

        self.fc1 = nn.Linear(3, 8) #Første lag med 3 input features og 8 neuroner
        self.fc2 = nn.Linear(8,1) #Andet lag med 8 input features og 1 neuron (output lag)
    
    def forward(self, x): #Forward funktionen der definerer hvordan data kommer gennem netværket
        x = torch.tanh(self.fc1(x)) #Aktiveringsfunktionen tanh anvendes efter første lag, bruges så outputtet er mellem -1 og 1 og ikke er lineært
        x = torch.sigmoid(self.fc2(x)) #Aktiveringsfunktionen sigmoid anvendes efter andet lag, bruges så outputtet er mellem 0 og 1 så det ikke bliver lineært
        return x
    
def should_flap(net, state): #Funktionen bestemmer om fuglen skal flap eller ej baseret på netværkets output
    with torch.no_grad(): #Deaktiverer gradient beregning for at spare hukommelse og computation
        output = net(state)
    return output.item() > 0.5 #Returnerer True hvis output er større end 0.5, ellers False

def create_population():
    return [FlappyNet() for _ in range(POP_SIZE)] #Opretter en liste med POP_SIZE antal FlappyNet (neurale netværk)

def get_genome(net):
    return torch.cat([net.fc1.weight.flatten(), net.fc1.bias, net.fc2.weight.flatten(), net.fc2.bias]) #Her Kombineres alle vægte og bias'er i et enkelt tensor, hvor en tensor er en multi-dimensionel matri

def set_genome(net, genome):

    i = 0 #Dette gør at vi kan holde styr på vores position i genome tensoren og bruges til at sikre at de rigtige vægte og bias'er bliver sat korrekt

    w1 = net.fc1.weight.numel() #Dette er antal elementer i vægtene for fc1 laget (.numel() returnerer antal elementer i en tensor.
    net.fc1.weight.data = genome[i:i+w1].view_as(net.fc1.weight) #Sætter vægtene for fc1 laget ved at tage en slice af genome tensoren og ændre dens form til at matche fc1 vægtene
    i += w1 #Opdaterer positionen i genome tensoren

    b1 = net.fc1.bias.numel() #Antal elementer i bias for fc1 laget (.numel() returnerer antal elementer i en tensor)
    net.fc1.bias.data = genome[i:i+b1] #Denne sætter bias for fc1 laget #Og net. gør at vi ændrer data attributten direkte (altså læs en variabel på objektet)
    i += b1

    w2 = net.fc2.weight.numel() #Antal elementer i vægtene for fc2 laget
    net.fc2.weight.data = genome[i:i+w2].view_as(net.fc2.weight) #Sætter vægtene for fc2 laget
    i += w2

    b2 = net.fc2.bias.numel()
    net.fc2.bias.data = genome[i:i+b2] #Sætter bias for fc2 laget (og [i:i+b2] tager en slice af genome tensoren)
    i += b2

def evaluate_fitness(net):
    H, W = 600, 450 #Højde og bredde på spilvinduet
    bird = Bird(100, H//2) #Opretter en fugl ved position (100, H/2)

    tunnels = [Tunnel.random(W + i*250, 70, 150, H, margin=60, speed=3) for i in range(3)] #Opretter en liste med 3 tilfældige tunneler, der er jævnt fordelt i x-aksen

    score, frames = 0, 0
    flaps = 0 

    while bird.alive and frames < MAX_FRAMES: #Kører spillet indtil fuglen dør eller maks antal frames er nået
        frames += 1

        t = min([t for t in tunnels if t.x + t.width > bird.x], key=lambda t: t.x) #Finder den nærmeste tunnel foran fuglen ved at filtrere tunnelerne og vælge den med den mindste x-værdi
        gap_center = t.gap_y + t.gap_height / 2 #Beregner midten af hullet i tunnelen

        state = torch.tensor([bird.y / H, bird.vel / bird.max_fall_speed, (gap_center - bird.y) / H], dtype=torch.float32) #Normaliserer input værdierne til netværket (bird.y, bird.vel, afstand til hullets center) ved at dividere med H og max_fall_speed

        if should_flap(net, state):
            bird.flap()
            flaps += 1

        bird.update()

        for t in tunnels: #Opdaterer hver tunnel i tunnelerne
            t.update()
            if not t.passed and t.x + t.width < bird.x: #Tjekker om tunnelen er passeret af fuglen
                t.passed, score = True, score + 1 #Hvis ja, opdateres passed til True og scoren øges med 1
            if t.is_offscreen(): #Tjekker om tunnelen er udenfor skærmen
                t.reset(max(tt.x for tt in tunnels) + 250) #Er dette tilfældet, vil tunnelen blive reset til en ny position indenfor skærmen
            if t.collides(bird): #Tjekker om fuglen kolliderer med tunnelen
                bird.alive = False #Hvis ja, sættes bird.alive til False

        if bird.y < 0 or bird.y > H:
            bird.alive = False

    return frames + (score * 1000), flaps #Returnerer fitness score (baseret på antal frames og score) og antal flaps og at gange med 1000 gør at score vægter mere i fitness evalueringen og kan evt ændres til at gøre den mere eller mindre vigtig

def mutate(genome, MUT_RATE, SCALE):
    for i in range(len(genome)): #Kører gennem alle gener i genomet
        if random.random() < MUT_RATE: #Undersøger om et tilfældigt tal er mindre end mutationsraten
            genome[i] += SCALE * torch.randn(()) #Hvis ja, tilføjes en tilfældig værdi (fra en normalfordeling) skaleret med SCALE til genet (Hvor torch.randn(1) genererer et tilfældigt tal fra en standard normalfordeling)
    return genome


def run_generation(population):
    scored_population = [] #Opretter en tom liste til at holde netværk og deres fitness score

#1 evaluerer fitness for hvert netværk i populationen
    for net in population:
        fitness, flaps = evaluate_fitness(net) #Evaluerer fitness for hvert netværk ved at kalde evaluate_fitness funktionen der simulerer et spil Flappy Bird
        scored_population.append((net, fitness, flaps)) #Tilføjer en tuple af netværket og dets fitness score til scored_population listen
    
#2 sorterer populationen baseret på fitness score

    scored_population.sort(key=lambda x:x[1], reverse=True) #Sorterer listen baseret på fitness score i faldende rækkefølge og lambda funktionen bruges til at specificere at sorteringen skal ske baseret på det andet element i tuplen (fitness score)
    best_fitness = scored_population[0][1]
    best_flaps = scored_population[0][2]
    avg_fitness = sum(x[1] for x in scored_population) / len(scored_population)
    print(f"Bedste fitness: {best_fitness}, Gennemsnitlig fitness: {avg_fitness}| Best flaps: {best_flaps}")

#3 Behold top 20% af populationen, altså "eliten"

    elite_count = max(1, int(0.02*POP_SIZE)) #ATallet 0,02 udvælger de 2% bedste af de 100 fugle #Beregner antal netværk der skal bevares som elite (mindst 1) og max funktionen sikrer at der altid er mindst 1 elite netværk
    elites = scored_population[:elite_count] #Beholder de top elite_count netværk fra den sorterede liste ved at slice listen

    new_population = [] #Opretter en ny tom liste til den nye population

#4 Kopierer elite netværkene direkte til den nye population

    for net, _, _ in elites: #Her bruges , til at ignorere den første værdi i hver tuple (altså fitness scoren) og kun fokusere på netværket
        elite_copy = FlappyNet()                          # nyt netværk
        set_genome(elite_copy, get_genome(net).clone())  # kopier vægte og bias 
        new_population.append(elite_copy) #Tilføjer kopien af elite netværket til den nye population

#5 Fylder resten af populationen med muterede kopier af elite netværkene

    while len(new_population) < POP_SIZE: #Kører indtil den nye population er den ønskede størrelse
        parent = random.choice(elites)[0] #Vælger tilfældigt et netværk fra eliten som forælder (her bruges [1] for at få netværket fra tuplen)
        child = FlappyNet() #Opretter et nyt netværk, der er "barnet"

        genome = get_genome(parent).clone() #Henter genomet for forælderen ved at kalde get_genome funktionen og .clone() laver en kopi af genomet så vi ikke ændrer på forælderens genom direkte
        genome = mutate(genome, MUT_RATE, SCALE) #Muterer genomet ved at kalde mutate funktionen

        set_genome(child, genome) #Sætter det muterede genom til barnet ved at kalde set_genome funktionen
        new_population.append(child) #Tilføjer barnet til den nye population
    
    return new_population #Returnerer den nye population

population = create_population() #Opretter den initiale population af neurale netværk

for gen in range(GENERATIONS): #Kører gennem det ønskede antal generationer
    population = run_generation(population) #Kører en generation ved at kalde run_generation funktionen og opdaterer populationen
    print(f"Generation {gen} færdig")
