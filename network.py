import dynetx as dn
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
g=nx.Graph()


read_data = pd.read_csv("First_Data_set.csv",  sep=',')
Data_set =pd.DataFrame(read_data)
df_Graph_node = Data_set.set_index(['unit_id']).stack().rename('user_id').reset_index().query('user_id != 0')
G = nx.from_pandas_edgelist(df_Graph_node,'level_1','unit_id')
print('done!')
D,T = nx.bipartite.sets(G)
pos = dict()
pos.update( (n, (1, i)) for i, n in enumerate(D) ) 
pos.update( (n, (2, i)) for i, n in enumerate(T) ) 
nx.draw(G, pos=pos, alpha=.4)
print('done!2')
for i in pos:
    x, y = pos[i]
    plt.text(x-.05, y+.2, i)

print('done!3')
plt.savefig("Graph.png", format="PNG")
plt.show()