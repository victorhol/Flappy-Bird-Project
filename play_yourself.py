import pygame
import sys
from Flappy_bird_Game import Tunnel, Bird

def main():
	SCREEN_WIDTH = 450
	global SCREEN_HEIGHT
	SCREEN_HEIGHT = 600
	FPS = 60

	pygame.init()
	screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
	pygame.display.set_caption("Flappy Bird - Simple")
	clock = pygame.time.Clock()

	font = pygame.font.SysFont(None, 36)

	# create player
	bird = Bird(100, SCREEN_HEIGHT // 2)

	# tunnels configuration
	gap_height = 150
	tunnel_width = 70
	tunnel_speed = 3
	spacing = 250
	num_tunnels = 3
	tunnels = [Tunnel.random(SCREEN_WIDTH + i * spacing, tunnel_width, gap_height, SCREEN_HEIGHT, margin=60, speed=tunnel_speed) for i in range(num_tunnels)]

	score = 0
	running = True

	def reset_game():
		nonlocal score, bird, tunnels
		score = 0
		bird.x = 100.0
		bird.y = float(SCREEN_HEIGHT // 2)
		bird.vel = 0.0
		bird.alive = True

		# reposition tunnels
		tunnels = [Tunnel.random(SCREEN_WIDTH + i * spacing, tunnel_width, gap_height, SCREEN_HEIGHT, margin=60, speed=tunnel_speed) for i in range(num_tunnels)]

	# main loop
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					if bird.alive:
						bird.flap()
					else:
						reset_game()
				elif event.key == pygame.K_r and not bird.alive:
					reset_game()
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if bird.alive:
					bird.flap()
				else:
					reset_game()

		# update
		if bird.alive:
			bird.update()

			for t in tunnels:
				t.update()

				# scoring: passed tunnel
				if not t.passed and t.x + t.width < bird.x:
					t.passed = True
					score += 1

				# recycle offscreen tunnels
			for t in tunnels:
				if t.is_offscreen():
					# place to right of current farthest tunnel
					rightmost_x = max(tt.x for tt in tunnels)
					t.reset(rightmost_x + spacing)

			# collisions
			for t in tunnels:
				if t.collides(bird):
					bird.alive = False
					break

			# boundaries
			if bird.y - bird.radius < 0 or bird.y + bird.radius > SCREEN_HEIGHT:
				bird.alive = False

		# draw
		screen.fill((135, 206, 235))  # sky blue

		for t in tunnels:
			t.draw(screen)

		bird.draw(screen)

		score_surf = font.render(str(score), True, (0, 0, 0))
		screen.blit(score_surf, (10, 10))

		if not bird.alive:
			over_surf = font.render("Game Over - Press Space to restart", True, (255, 0, 0))
			rect = over_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
			screen.blit(over_surf, rect)

		pygame.display.flip()
		clock.tick(FPS)

	pygame.quit()
	sys.exit()


if __name__ == "__main__":
	main()

