# from asyncio import loop
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from math import sin, cos, atan2, pi
from IPython.display import display, Math, Latex, Markdown, HTML

print(5)

def plot_data(data_1, data_2, label_1, label_2, markersize_1=8, markersize_2=8):
  fig = plt.figure(figsize=(10, 6))
  ax = fig.add_subplot(111)
  ax.axis('equal')
  if data_1 is not None:
      x_p, y_p = data_1
      ax.plot(x_p, y_p, color='#336699', markersize=markersize_1, marker='o', linestyle=":", label=label_1)
  if data_2 is not None:
      x_q, y_q = data_2
      ax.plot(x_q, y_q, color='orangered', markersize=markersize_2, marker='o', linestyle=":", label=label_2)
  ax.legend()
  return ax

def plot_values(values, label):
  fig = plt.figure(figsize=(10, 4))
  ax = fig.add_subplot(111)
  ax.plot(values, label=label)
  ax.legend()
  ax.grid(True)
  plt.show()
    
def animate_results(P_values, Q, corresp_values, xlim, ylim):
  """A function used to animate the iterative processes we use."""
  fig = plt.figure(figsize=(10, 6))
  anim_ax = fig.add_subplot(111)
  anim_ax.set(xlim=xlim, ylim=ylim)
  anim_ax.set_aspect('equal')
  plt.close()
  x_q, y_q = Q
  # draw initial correspondeces
  corresp_lines = []
  for i, j in correspondences:
      corresp_lines.append(anim_ax.plot([], [], 'grey')[0])
  # Prepare Q data.
  Q_line, = anim_ax.plot(x_q, y_q, 'o', color='orangered')
  # prepare empty line for moved data
  P_line, = anim_ax.plot([], [], 'o', color='#336699')

  def animate(i):
      P_inc = P_values[i]
      x_p, y_p = P_inc
      P_line.set_data(x_p, y_p)
      draw_inc_corresp(P_inc, Q, corresp_values[i])
      return (P_line,)
  
  def draw_inc_corresp(points_from, points_to, correspondences):
      for corr_idx, (i, j) in enumerate(correspondences):
          x = [points_from[0, i], points_to[0, j]]
          y = [points_from[1, i], points_to[1, j]]
          corresp_lines[corr_idx].set_data(x, y)
  
  anim = animation.FuncAnimation(fig, animate,
                                  frames=len(P_values), 
                                  interval=500, 
                                  blit=True)
  return HTML(anim.to_jshtml())

# Source frame: Q, Target frame: P
def get_correspondence_indices(P, Q):
  """For each point in P find closest one in Q."""
  p_size = P.shape[1]
  q_size = Q.shape[1]
  correspondences = []
  for i in range(p_size):
      p_point = P[:, i]
      # if i == 1:
      #   print(sys.maxsize)
      min_dist = sys.maxsize
      chosen_idx = -1
      for j in range(q_size):
          q_point = Q[:, j]
          dist = np.linalg.norm(q_point - p_point)
          if dist < min_dist:
              min_dist = dist
              chosen_idx = j
      correspondences.append((i, chosen_idx))
  return correspondences

def draw_correspondeces(P, Q, correspondences, ax):
  label_added = False
  for i, j in correspondences:
      x = [P[0, i], Q[0, j]]
      y = [P[1, i], Q[1, j]]
      if not label_added:
          ax.plot(x, y, color='grey', label='correpondences')
          label_added = True
      else:
          ax.plot(x, y, color='grey')
  ax.legend()
  
def center_data(data, exclude_indices=[]):
  exclude_indices = [10]
  reduced_data = np.delete(data, exclude_indices, axis=1)
  # print("Reduced: \n{}".format(reduced_data))
  # print(reduced_data.shape)
  center = np.array([reduced_data.mean(axis=1)]).T
  # print(center.shape)
  # print(data -center)
  return center, data - center

def compute_cross_covariance(P, Q, correspondences, kernel=lambda diff: 1.0):
  cov = np.zeros((2, 2))
  exclude_indices = []
  loop_count = 0;
  for i, j in correspondences:
      p_point = P[:, [i]]
      q_point = Q[:, [j]]
      weight = kernel(p_point - q_point)
      if weight < 0.01: exclude_indices.append(i)
      cov += weight * q_point.dot(p_point.T)
      
      loop_count += 1
      # print(weight)
  # print(cov)
  cov = cov / loop_count
  # print(cov)
  return cov, exclude_indices

def icp_svd(P, Q, iterations=10, kernel=lambda diff: 1.0):
  """Perform ICP using SVD."""
  center_of_Q, Q_centered = center_data(Q)
  norm_values = []
  P_values = [P.copy()]
  P_copy = P.copy()
  corresp_values = []
  exclude_indices = []
  for i in range(iterations):
      center_of_P, P_centered = center_data(P_copy, exclude_indices=exclude_indices)
      correspondences = get_correspondence_indices(P_centered, Q_centered)
      corresp_values.append(correspondences)
      norm_values.append(np.linalg.norm(P_centered - Q_centered))
      cov, exclude_indices = compute_cross_covariance(P_centered, Q_centered, correspondences, kernel)
      U, S, V_T = np.linalg.svd(cov)
      R = U.dot(V_T)  
      t = center_of_Q - R.dot(center_of_P)  
      P_copy = R.dot(P_copy) + t
      
      # ax = plot_data(P_copy, Q, label_1='P_copy', label_2='Q')
      
      P_values.append(P_copy)
  corresp_values.append(corresp_values[-1])
  return P_values, norm_values, corresp_values

# initialize pertrubation rotation
angle = pi / 6
R_true = np.array([[cos(angle), -sin(angle)], 
                   [sin(angle),  cos(angle)]])
t_true = np.array([[-2], [5]])

# Generate data as a list of 2d points
num_points = 30
true_data = np.zeros((2, num_points))
true_data[0, :] = range(0, num_points)
# true_data[1, :] = 0.2 * true_data[0, :]* np.sin(0.5 * true_data[0, :]) 
true_data[1, :] = 0.2 * true_data[0,:] * np.sin(0.5* true_data[0,:])
# print(true_data)
# Move the data // S' = R*S + T
moved_data = R_true.dot(true_data) + t_true

# Assign to variables we use in formulas.
Q = true_data
P = moved_data
# plot_data(moved_data, true_data, "P: moved data", "Q: true data")
# plt.show()

# print("P: {}".format(P))
center_of_P, P_centered = center_data(P)
center_of_Q, Q_centered = center_data(Q)
# ax = plot_data(P_centered, Q_centered,
#                label_1='Moved data centered',
#                label_2='True data centered')
# plt.show()

correspondences = get_correspondence_indices(P_centered, Q_centered)
# print(correspondences)
# ax = plot_data(P_centered, Q_centered,
#                label_1='P centered',
#                label_2='Q centered')
# draw_correspondeces(P_centered, Q_centered, correspondences, ax)
# plt.show()

cov, _ = compute_cross_covariance(P_centered, Q_centered, correspondences)
# print(cov)

U, S, V_T = np.linalg.svd(cov)
# print(U)   R  : orthogonal mat.
# print(S)   S  : vector
# print(V_T) V_T: orthogonal mat.
R_found = U.dot(V_T)
R_found_reverse = V_T.dot(U)
print("R_found =\n", R_found)
print("R_reverse =\n", R_found_reverse)
t_found = center_of_Q - R_found.dot(center_of_P)
# print("R_found =\n", R_found)s
# print("t_found =\n", t_found)

P_corrected = R_found.dot(P) + t_found


P_corrected = R_found.dot(P) + t_found
# ax = plot_data(P_corrected, Q, label_1='P corrected', label_2='Q')
# plt.show()
# print("Squared diff: (P_corrected - Q) = ", np.linalg.norm(P_corrected - Q))

P_values, norm_values, corresp_values = icp_svd(P, Q)
plot_values(norm_values, label="Squared diff P->Q")
# ax = plot_data(P_values[-1], Q, label_1='P final', label_2='Q', markersize_1=15)
# plt.show()
# print(norm_values)

# animate_results(P_values, Q, corresp_values, xlim=(-5, 35), ylim=(-5, 35))
# plt.show()