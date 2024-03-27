from gridworld import GridWorld
from policy import policy
import matplotlib.pyplot as plt

# Créer une instance de Grid World
mdp_instance = GridWorld(noise=0.2, width=3, height=3, initial_state=(0, 0), goals={(2, 2): 1}, blocked_states=[(1, 1)], action_cost=-0.04)

# Créer une instance de votre politique avec l'instance Grid World comme argument
policy_instance = policy({} ,mdp_instance)

state = mdp_instance.get_initial_state()
plt.ion()  # Active le mode interactif de Matplotlib

while not mdp_instance.is_terminal(state):
    # Affiche la grille avec la position actuelle de l'agent
    plt.clf()  # Nettoie la figure courante pour préparer la nouvelle visualisation
    mdp_instance.visualise(agent_position=state)  # Visualisez l'état actuel avec la position de l'agent
    plt.draw()  # Mise à jour de la fenêtre graphique avec le contenu dessiné
    plt.pause(0.1)  # Une courte pause pour s'assurer que la fenêtre graphique se met à jour

    input("Appuyez sur Entrée pour faire avancer l'agent...")  # Attendre que l'utilisateur appuie sur Entrée
    action = policy_instance.select_action(state)
    new_state, reward = mdp_instance.execute(state, action)
    
    # Mise à jour de l'état pour la prochaine itération
    state = new_state

plt.ioff()  # Désactive le mode interactif
plt.show()  # Affiche la fenêtre graphique si nécessaire
