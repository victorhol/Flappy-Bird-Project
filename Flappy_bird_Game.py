import random
import pygame


class Bird:
	def __init__(self, x, y, radius=12):
		#birdies koordinater når den bliver "spawnet"
		self.x = float(x)
		self.y = float(y)
		
        #størrelsen på birdie
		self.radius = int(radius)
		
        #udgangspunkt for fald upåvirket af gravity og hop
		self.vel = 0.0
		
        #hvor hurtigt falder birdie
		self.gravity = 0.5
		self.max_fall_speed = 12.0
		
        #hvor højt hopper birdie
		self.flap_strength = -8.5        
		
        #om den lever
		self.alive = True

    # flap funktionen sætter bare -8.5 til velocity, for et midlertidigt updraft
	def flap(self):
		self.vel = self.flap_strength

    # udviklingen af birdies rætning og hastighed, og rykker hvor birdie er i envionmentet
	def update(self):
		self.vel += self.gravity
		if self.vel > self.max_fall_speed:
			self.vel = self.max_fall_speed
		self.y += self.vel

    #hitbox for birdie
	def get_rect(self):
		return pygame.Rect(int(self.x - self.radius), int(self.y - self.radius), self.radius * 2, self.radius * 2)

    #tegner birdie på mappet ved dens nye coordinator
	def draw(self, surface, color=(255, 255, 0)):
		pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.radius)


class Tunnel:
	def __init__(self, x, width, gap_y, gap_height, screen_height, speed=3):
		#bredde, højde, mellemrum
		self.x = float(x)
		self.width = int(width)
		self.gap_y = int(gap_y)
		self.gap_height = int(gap_height)
		
        #Skærmhøjde(nok ikke nødvendig her)
		self.screen_height = int(screen_height)
		
		#hastigheden tunnellerne rykker mod birdie
		self.speed = float(speed)
		
        #er tunnellen passeret (starter som nej)
		self.passed = False

	#En constructor som laver tunnellerne random
	@classmethod
	def random(cls, x, width, gap_height, screen_height, margin=50, speed=3):
		gap_y = random.randint(margin, screen_height - margin - gap_height)
		return cls(x, width, gap_y, gap_height, screen_height, speed)

	#rykker birdie
	def update(self):
		self.x -= self.speed

	#er birdie på skærmen
	def is_offscreen(self):
		return self.x + self.width < 0

	#resetter spillet
	def reset(self, new_x):
		self.x = float(new_x)
		self.passed = False
		self.gap_y = random.randint(50, self.screen_height - 50 - self.gap_height)

	#hitbox for top tunnellen
	def top_rect(self):
		return pygame.Rect(int(self.x), 0, self.width, int(self.gap_y))

	#hitbox for bottom tunnel
	def bottom_rect(self):
		top_of_bottom = int(self.gap_y + self.gap_height)
		return pygame.Rect(int(self.x), top_of_bottom, self.width, int(self.screen_height - top_of_bottom))

	# tegner tunnellerne
	def draw(self, surface, color=(0, 200, 0)):
		pygame.draw.rect(surface, color, self.top_rect())
		pygame.draw.rect(surface, color, self.bottom_rect())

	#undersøger om birdie har ramt tunnellen
	def collides(self, bird: Bird) -> bool:
		brect = bird.get_rect()
		return self.top_rect().colliderect(brect) or self.bottom_rect().colliderect(brect)
