#Import af pakker
import torch
import torch.nn as nn
import random
import argparse

try:
    import pandas as pd
except ImportError:  # pandas is optional at runtime
    pd = None
from Flappy_bird_Game import Bird, Tunnel

POP_SIZE = 100 #Størrelse på populationen
GENERATIONS = 40 #Antal generationer
RUNS = 20 #Antal runs (hele træningsforløb)
MUT_RATE = 0.1 #Mutationsrate
SCALE = 0.2 #Hvor meget mutationerne skal påvirke genomet
TARGET_PIPES = 999 #Stopper træning når bedste score (rør) når denne

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

    while bird.alive:
        frames += 1

        t = min([t for t in tunnels if t.x + t.width > bird.x], key=lambda t: t.x) #Finder den nærmeste tunnel foran fuglen ved at filtrere tunnelerne og vælge den med den mindste x-værdi
        gap_center = t.gap_y + t.gap_height / 2 #Beregner midten af hullet i tunnelen

        state = torch.tensor([bird.y / H, bird.vel / bird.max_fall_speed, (gap_center - bird.y) / H], dtype=torch.float32) #Normaliserer input værdierne til netværket (bird.y, bird.vel, afstand til hullets center) ved at dividere med H og max_fall_speed

        if should_flap(net, state):
            bird.flap()

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

    pipes = min(score, TARGET_PIPES)
    return frames + (pipes * 500), pipes #Returnerer fitness score og capped antal rør (pipes)

def mutate(genome, MUT_RATE, SCALE):
    for i in range(len(genome)): #Kører gennem alle gener i genomet
        if random.random() < MUT_RATE: #Undersøger om et tilfældigt tal er mindre end mutationsraten
            genome[i] += SCALE * torch.randn(()) #Hvis ja, tilføjes en tilfældig værdi (fra en normalfordeling) skaleret med SCALE til genet (Hvor torch.randn(1) genererer et tilfældigt tal fra en standard normalfordeling)
    return genome


def run_generation(population, elite_n):
    scored_population = [] #Opretter en tom liste til at holde netværk og deres fitness score

#1 evaluerer fitness for hvert netværk i populationen
    for net_idx, net in enumerate(population):
        fitness, pipes = evaluate_fitness(net) #Evaluerer fitness for hvert netværk ved at kalde evaluate_fitness funktionen der simulerer et spil Flappy Bird
        scored_population.append((net, fitness, pipes, net_idx)) #Tilføjer en tuple af netværket, fitness score, antal rør (pipes) og index
    
#2 sorterer populationen baseret på fitness score

    scored_population.sort(key=lambda x:x[1], reverse=True) #Sorterer listen baseret på fitness score i faldende rækkefølge og lambda funktionen bruges til at specificere at sorteringen skal ske baseret på det andet element i tuplen (fitness score)
    best_fitness = scored_population[0][1]
    best_pipes = min(scored_population[0][2], TARGET_PIPES)
    avg_fitness = sum(x[1] for x in scored_population) / len(scored_population)
    avg_pipes = sum(x[2] for x in scored_population) / len(scored_population)
    


    per_net_rows = [
        {"net_idx": x[3], "fitness": float(x[1]), "pipes": int(x[2])}
        for x in scored_population
    ]

#3 Behold top 5% af populationen, altså "eliten"

    elite_count = elite_n #ATallet 0,05 udvælger de 5% bedste af de 100 fugle #Beregner antal netværk der skal bevares som elite (mindst 1) og max funktionen sikrer at der altid er mindst 1 elite netværk
    elites = scored_population[:elite_count] #Beholder de top elite_count netværk fra den sorterede liste ved at slice listen

    new_population = [] #Opretter en ny tom liste til den nye population

#4 Kopierer elite netværkene direkte til den nye population

    for net, _, _, _ in elites: #Her bruges , til at ignorere nogle værdier i hver tuple og kun fokusere på netværket
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
    
    gen_row = {
        "best_fitness": float(best_fitness),
        "avg_fitness": float(avg_fitness),
        "best_pipes": int(best_pipes),
        "avg_pipes": float(avg_pipes),
    }
    return new_population, gen_row, per_net_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument("--summary_csv", type=str, default="ga_summary.csv")
    parser.add_argument("--details_csv", type=str, default="ga_details.csv")
    parser.add_argument("--elite_n", type=int, default=5, help="Antal elite-agenter (Top X)")

    args = parser.parse_args()

    if pd is None:
        raise SystemExit("pandas er ikke installeret. Kør: pip install pandas")

    summary_rows = []
    details_rows = []

    for run_idx in range(1, args.runs + 1):
        random.seed(1000 + run_idx)
        torch.manual_seed(1000 + run_idx)

        population = create_population() #Opretter den initiale population af neurale netværk
        run_highscore = 0
        run_best_fitness = float("-inf")

        for gen in range(1, args.generations + 1):
            population, gen_row, per_net = run_generation(population, args.elite_n)
            print(
                f"Generation {gen}/{args.generations} | "
                f"best_fitness={gen_row['best_fitness']:.2f} | "
                f"best_pipes={gen_row['best_pipes']}"
            )
            run_highscore = max(run_highscore, gen_row["best_pipes"])
            run_best_fitness = max(run_best_fitness, gen_row["best_fitness"])

            summary_rows.append({
                "run": run_idx,
                "generation": gen,
                "avg_fitness": gen_row["avg_fitness"],
                "best_fitness": gen_row["best_fitness"],
                "avg_pipes": gen_row["avg_pipes"],
                "highscore_pipes": gen_row["best_pipes"],
                "highscore_pipes_so_far": run_highscore,
                "best_fitness_so_far": run_best_fitness,
            })

            for r in per_net:
                r.update({"run": run_idx, "generation": gen})
                details_rows.append(r)

            """if gen_row["best_pipes"] >= TARGET_PIPES:
                print(f"Run {run_idx}: stopper ved {TARGET_PIPES} pipes i generation {gen}")
                break"""

    pd.DataFrame(summary_rows).to_csv(args.summary_csv, index=False)
    pd.DataFrame(details_rows).to_csv(args.details_csv, index=False)
    print(f"Skrev {args.summary_csv} og {args.details_csv}")

if __name__ == "__main__":
    main()

