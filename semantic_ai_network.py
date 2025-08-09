import json
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set

import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------
# Modelos de la Red Semántica
# ---------------------------

class Relationship:
    """
    Tipo de relación semántica. Usa valores estándar para facilitar filtrado/colores.
    """
    ISA = "es_un"                 # IS-A / Hiponimia (X es un Y)
    PART_OF = "parte_de"          # Meronimia (X es parte de Y)
    KIND_OF = "tipo_de"           # Similar a es_un pero útil para taxonomía informal
    ASSOCIATED_WITH = "asociado_con"  # Asociación general
    USED_FOR = "usado_para"       # Función/propósito
    REQUIRES = "requiere"         # Dependencia
    PRODUCES = "produce"          # Resultado/efecto
    CONTRASTS_WITH = "contrasta_con"  # Opuesto/comparado
    DERIVED_FROM = "derivado_de"  # Derivación

    @staticmethod
    def all():
        return {
            Relationship.ISA, Relationship.PART_OF, Relationship.KIND_OF,
            Relationship.ASSOCIATED_WITH, Relationship.USED_FOR, Relationship.REQUIRES,
            Relationship.PRODUCES, Relationship.CONTRASTS_WITH, Relationship.DERIVED_FROM
        }


class SemanticNetwork:
    """
    Red semántica dirigida (MultiDiGraph) con utilidades para:
    - añadir conceptos y relaciones
    - cargar/guardar JSON
    - consultar vecindarios por profundidad
    - visualizar y exportar PNG
    """
    def __init__(self):
        self.G = nx.MultiDiGraph()  # Permite múltiples aristas entre dos nodos con etiquetas distintas

    # --- CRUD Conceptos/Relaciones ---

    def add_concept(self, concept: str, **attrs):
        concept = concept.strip()
        if concept and concept not in self.G:
            self.G.add_node(concept, **attrs)

    def add_relation(self, src: str, rel: str, dst: str, **attrs):
        if rel not in Relationship.all():
            raise ValueError(f"Relación '{rel}' no es válida. Usa una de: {sorted(Relationship.all())}")
        self.add_concept(src)
        self.add_concept(dst)
        self.G.add_edge(src, dst, label=rel, **attrs)

    # --- Persistencia ---

    def to_json(self) -> Dict:
        data = {
            "nodes": [{"id": n, **self.G.nodes[n]} for n in self.G.nodes()],
            "edges": [{"src": u, "dst": v, **d} for u, v, k, d in self.G.edges(keys=True, data=True)]
        }
        return data

    @staticmethod
    def from_json(data: Dict) -> "SemanticNetwork":
        sn = SemanticNetwork()
        for n in data.get("nodes", []):
            nid = n.pop("id")
            sn.G.add_node(nid, **n)
        for e in data.get("edges", []):
            src, dst = e.pop("src"), e.pop("dst")
            sn.G.add_edge(src, dst, **e)
        return sn

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: str) -> "SemanticNetwork":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SemanticNetwork.from_json(data)

    # --- Consulta ---

    def neighborhood(self, center: str, depth: int = 1, directions: str = "both") -> Set[str]:
        """
        Retorna el conjunto de nodos a distancia <= depth desde 'center'.
        directions: 'out', 'in' o 'both'
        """
        if center not in self.G:
            return set()
        visited = {center}
        q = deque([(center, 0)])
        while q:
            node, dist = q.popleft()
            if dist == depth:
                continue
            neighbors = set()
            if directions in ("both", "out"):
                neighbors |= set(self.G.successors(node))
            if directions in ("both", "in"):
                neighbors |= set(self.G.predecessors(node))
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))
        return visited

    def relations_from_to(self, src: str, dst: str) -> List[str]:
        if not (src in self.G and dst in self.G):
            return []
        rels = []
        for _, _, d in self.G.edges(src, dst, data=True):
            rels.append(d.get("label", ""))
        return rels

    def subgraph_for(self, center: str, depth: int = 1) -> nx.MultiDiGraph:
        nodes = self.neighborhood(center, depth=depth, directions="both")
        return self.G.subgraph(nodes).copy()

    def explain_text(self, center: str, depth: int = 1) -> str:
        """
        Imprime relaciones salientes y entrantes del vecindario.
        """
        if center not in self.G:
            return f"❌ El concepto '{center}' no existe en la red."
        sub = self.subgraph_for(center, depth)
        lines = []
        lines.append(f"🔎 Concepto: {center} (profundidad {depth})\n")

        # Salientes
        out_map = defaultdict(list)
        for u, v, d in sub.edges(data=True):
            if u == center:
                out_map[d.get("label", "relación")].append(v)

        if out_map:
            lines.append("➡ Relaciones *desde* el concepto:")
            for rel, dests in sorted(out_map.items()):
                dests = sorted(set(dests))
                lines.append(f"  - {rel} → {', '.join(dests)}")
            lines.append("")

        # Entrantes
        in_map = defaultdict(list)
        for u, v, d in sub.edges(data=True):
            if v == center:
                in_map[d.get("label", "relación")].append(u)

        if in_map:
            lines.append("⬅ Relaciones *hacia* el concepto:")
            for rel, srcs in sorted(in_map.items()):
                srcs = sorted(set(srcs))
                lines.append(f"  - {', '.join(srcs)} → {rel}")
            lines.append("")

        # Otros enlaces en el vecindario (no centrados)
        other_edges = []
        for u, v, d in sub.edges(data=True):
            if u != center and v != center:
                other_edges.append((u, d.get("label", "relación"), v))
        if other_edges:
            lines.append("🔗 Otras conexiones en el vecindario:")
            for u, rel, v in sorted(other_edges):
                lines.append(f"  - {u} —{rel}→ {v}")

        return "\n".join(lines)

    # --- Visualización ---

    def draw(self, center: str, depth: int = 1, save_path: Optional[str] = None, dpi: int = 140):
        """
        Visualiza (y opcionalmente guarda como PNG) el subgrafo alrededor de 'center'.
        """
        if center not in self.G:
            print(f"❌ El concepto '{center}' no existe en la red.")
            return

        sub = self.subgraph_for(center, depth)
        pos = nx.spring_layout(sub, seed=42, k=0.6)  # Layout reproducible

        # Colores por tipo de relación
        rel_color = {
            Relationship.ISA:            "#1f77b4",
            Relationship.PART_OF:        "#2ca02c",
            Relationship.KIND_OF:        "#17becf",
            Relationship.ASSOCIATED_WITH:"#7f7f7f",
            Relationship.USED_FOR:       "#ff7f0e",
            Relationship.REQUIRES:       "#9467bd",
            Relationship.PRODUCES:       "#8c564b",
            Relationship.CONTRASTS_WITH: "#d62728",
            Relationship.DERIVED_FROM:   "#bcbd22",
        }

        # Dibujo nodos
        node_colors = []
        for n in sub.nodes():
            if n == center:
                node_colors.append("#FFD166")  # centro
            else:
                node_colors.append("#A8DADC")  # otros

        nx.draw_networkx_nodes(sub, pos, node_size=1400, node_color=node_colors, edgecolors="#333333")
        nx.draw_networkx_labels(sub, pos, font_size=9)

        # Dibujo aristas (por tipo de relación)
        # Como es MultiDiGraph, agrupar por etiqueta
        edges_by_label: Dict[str, List[Tuple[str, str, Dict]]] = defaultdict(list)
        for u, v, d in sub.edges(data=True):
            label = d.get("label", "")
            edges_by_label[label].append((u, v, d))

        for label, edges in edges_by_label.items():
            color = rel_color.get(label, "#444444")
            nx.draw_networkx_edges(
                sub, pos,
                edgelist=[(u, v) for (u, v, _) in edges],
                arrows=True, arrowstyle="-|>", width=2, edge_color=color, alpha=0.9
            )

        # Etiquetas para aristas (mostrar tipo de relación)
        edge_labels = {(u, v): d.get("label", "") for u, v, d in sub.edges(data=True)}
        nx.draw_networkx_edge_labels(sub, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

        plt.title(f"Red semántica: '{center}' (profundidad {depth})")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi)
            print(f"💾 Imagen guardada en: {save_path}")
            plt.close()
        else:
            plt.show()


# ---------------------------------
# Carga/Inicialización del Conocimiento
# ---------------------------------

def load_or_create_kb(path: str) -> SemanticNetwork:
    if os.path.exists(path):
        return SemanticNetwork.load_json(path)

    sn = SemanticNetwork()

    # --- Núcleo de IA ---
    sn.add_concept("Inteligencia Artificial")
    sn.add_concept("Aprendizaje Automático")
    sn.add_relation("Inteligencia Artificial", Relationship.KIND_OF, "Informática")
    sn.add_relation("Inteligencia Artificial", Relationship.ASSOCIATED_WITH, "Ética de IA")
    sn.add_relation("Inteligencia Artificial", Relationship.ASSOCIATED_WITH, "Robótica")
    sn.add_relation("Inteligencia Artificial", Relationship.ASSOCIATED_WITH, "Visión por Computador")
    sn.add_relation("Inteligencia Artificial", Relationship.ASSOCIATED_WITH, "Procesamiento de Lenguaje Natural")
    sn.add_relation("Inteligencia Artificial", Relationship.ASSOCIATED_WITH, "Sistemas Expertos")
    sn.add_relation("Inteligencia Artificial", Relationship.ASSOCIATED_WITH, "Planificación")
    sn.add_relation("Inteligencia Artificial", Relationship.ASSOCIATED_WITH, "Razonamiento Lógico")
    sn.add_relation("Inteligencia Artificial", Relationship.ASSOCIATED_WITH, "Representación del Conocimiento")
    sn.add_relation("Inteligencia Artificial", Relationship.ASSOCIATED_WITH, "Aprendizaje Automático")

    # --- ML (ramas) ---
    sn.add_relation("Aprendizaje Automático", Relationship.KIND_OF, "Inteligencia Artificial")
    sn.add_relation("Aprendizaje Automático", Relationship.ASSOCIATED_WITH, "Datos")
    sn.add_relation("Aprendizaje Automático", Relationship.REQUIRES, "Características (features)")
    sn.add_relation("Aprendizaje Automático", Relationship.PRODUCES, "Modelo")

    for sub in ["Aprendizaje Supervisado", "Aprendizaje No Supervisado", "Aprendizaje por Refuerzo", "Aprendizaje Semisupervisado", "Aprendizaje Auto-supervisado"]:
        sn.add_relation("Aprendizaje Automático", Relationship.PART_OF, sub)  # (ramas como 'parte_de' para graficar divergente)

    # --- Supervisado ---
    sn.add_relation("Aprendizaje Supervisado", Relationship.USED_FOR, "Clasificación")
    sn.add_relation("Aprendizaje Supervisado", Relationship.USED_FOR, "Regresión")
    sn.add_relation("Aprendizaje Supervisado", Relationship.REQUIRES, "Etiquetas")
    sn.add_relation("Clasificación", Relationship.ISA, "Tarea Supervisada")
    sn.add_relation("Regresión", Relationship.ISA, "Tarea Supervisada")

    # Algoritmos supervisados
    for alg in ["Regresión Logística", "SVM", "Árboles de Decisión", "Bosques Aleatorios", "k-NN", "Naive Bayes", "Redes Neuronales"]:
        sn.add_relation(alg, Relationship.ISA, "Algoritmo Supervisado")
        sn.add_relation("Aprendizaje Supervisado", Relationship.ASSOCIATED_WITH, alg)

    # Métricas supervisadas
    for m in ["Exactitud", "Precisión", "Exhaustividad (Recall)", "F1", "AUC-ROC", "MSE", "MAE", "RMSE"]:
        sn.add_relation(m, Relationship.ISA, "Métrica")
        sn.add_relation("Aprendizaje Supervisado", Relationship.USED_FOR, m)

    # --- No supervisado ---
    sn.add_relation("Aprendizaje No Supervisado", Relationship.USED_FOR, "Clustering")
    sn.add_relation("Aprendizaje No Supervisado", Relationship.USED_FOR, "Reducción de Dimensionalidad")
    sn.add_relation("Aprendizaje No Supervisado", Relationship.USED_FOR, "Detección de Anomalías")

    for alg in ["k-means", "DBSCAN", "GMM", "Jerárquico", "PCA", "t-SNE", "UMAP", "Autoencoders"]:
        sn.add_relation(alg, Relationship.ISA, "Algoritmo No Supervisado")
        sn.add_relation("Aprendizaje No Supervisado", Relationship.ASSOCIATED_WITH, alg)

    # --- Por Refuerzo ---
    sn.add_relation("Aprendizaje por Refuerzo", Relationship.REQUIRES, "Agente")
    sn.add_relation("Aprendizaje por Refuerzo", Relationship.REQUIRES, "Entorno")
    sn.add_relation("Aprendizaje por Refuerzo", Relationship.REQUIRES, "Recompensa")
    sn.add_relation("Aprendizaje por Refuerzo", Relationship.USED_FOR, "Toma de decisiones secuenciales")

    for alg in ["Q-Learning", "SARSA", "DQN", "PPO", "A3C"]:
        sn.add_relation(alg, Relationship.ISA, "Algoritmo de Refuerzo")
        sn.add_relation("Aprendizaje por Refuerzo", Relationship.ASSOCIATED_WITH, alg)

    # --- PLN ---
    sn.add_relation("Procesamiento de Lenguaje Natural", Relationship.PART_OF, "Inteligencia Artificial")
    for sub in ["Tokenización", "Lematización", "Análisis Sintáctico", "Embeddings", "Traducción Automática", "Resumen Automático", "Reconocimiento de Entidades", "Modelos de Lenguaje"]:
        sn.add_relation("Procesamiento de Lenguaje Natural", Relationship.ASSOCIATED_WITH, sub)

    for model in ["Word2Vec", "GloVe", "FastText", "ELMo", "BERT", "GPT", "T5", "mT5"]:
        sn.add_relation(model, Relationship.ISA, "Modelo de Lenguaje")
        sn.add_relation("Modelos de Lenguaje", Relationship.ASSOCIATED_WITH, model)

    # --- Visión ---
    sn.add_relation("Visión por Computador", Relationship.PART_OF, "Inteligencia Artificial")
    for sub in ["Clasificación de Imágenes", "Detección de Objetos", "Segmentación Semántica", "Estimación de Pose", "OCR"]:
        sn.add_relation("Visión por Computador", Relationship.ASSOCIATED_WITH, sub)

    for model in ["CNN", "ResNet", "VGG", "Inception", "EfficientNet", "ViT"]:
        sn.add_relation(model, Relationship.ISA, "Arquitectura de Red Neuronal")
        sn.add_relation("Visión por Computador", Relationship.ASSOCIATED_WITH, model)

    # --- Robótica / Planificación / Razonamiento ---
    sn.add_relation("Robótica", Relationship.PART_OF, "Inteligencia Artificial")
    sn.add_relation("Planificación", Relationship.PART_OF, "Inteligencia Artificial")
    sn.add_relation("Razonamiento Lógico", Relationship.PART_OF, "Inteligencia Artificial")
    sn.add_relation("Sistemas Expertos", Relationship.PART_OF, "Inteligencia Artificial")

    for sub in ["Lógica Proposicional", "Lógica de Primer Orden", "Inferencia", "Encadenamiento hacia atrás", "Encadenamiento hacia adelante"]:
        sn.add_relation("Razonamiento Lógico", Relationship.ASSOCIATED_WITH, sub)

    for sub in ["Adquisición del Conocimiento", "Ontologías", "Grafos de Conocimiento"]:
        sn.add_relation("Representación del Conocimiento", Relationship.ASSOCIATED_WITH, sub)

    # --- Ética ---
    for topic in ["Sesgo Algorítmico", "Privacidad", "Explicabilidad (XAI)", "Gobernanza de IA", "Seguridad"]:
        sn.add_relation("Ética de IA", Relationship.ASSOCIATED_WITH, topic)

    # --- Pipeline de ML ---
    for step in ["Recolección de Datos", "Limpieza de Datos", "Ingeniería de Características", "Entrenamiento", "Validación", "Evaluación", "Despliegue", "Monitoreo"]:
        sn.add_relation("Pipeline de ML", Relationship.PART_OF, step)
    sn.add_relation("Aprendizaje Automático", Relationship.ASSOCIATED_WITH, "Pipeline de ML")

    # Guardar KB inicial
    sn.save_json(path)
    return sn


# --------------
# Interfaz (CLI)
# --------------

def main():
    kb_path = "kb_ai.json"
    sn = load_or_create_kb(kb_path)

    print("=== Red Semántica de IA ===")
    print("Escribe un concepto (ej: 'Inteligencia Artificial', 'Aprendizaje Supervisado', 'BERT').")
    print("Comandos: ':add' para añadir, ':exit' para salir.\n")

    while True:
        concept = input("🔹 Concepto> ").strip()
        if not concept:
            continue
        if concept == ":exit":
            print("¡Hasta luego!")
            break
        if concept == ":add":
            print("Añadir relación: formato => origen | tipo_relación | destino")
            print(f"Tipos válidos: {sorted(Relationship.all())}")
            line = input("Relación> ").strip()
            try:
                src, rel, dst = [x.strip() for x in line.split("|")]
                sn.add_relation(src, rel, dst)
                sn.save_json(kb_path)
                print("✅ Relación añadida y KB guardada.\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")
            continue

        if concept not in sn.G:
            print(f"❌ '{concept}' no existe. Puedes añadirlo con ':add'.\n")
            continue

        try:
            depth = int(input("Profundidad (1-3 recomendado)> ").strip() or "1")
        except ValueError:
            depth = 1

        print("Elige salida: 1) Texto  2) Ver gráfico  3) Guardar PNG")
        choice = (input("Opción> ").strip() or "1")

        if choice == "1":
            print()
            print(sn.explain_text(concept, depth))
            print()
        elif choice == "2":
            sn.draw(concept, depth=depth, save_path=None)
        elif choice == "3":
            os.makedirs("output", exist_ok=True)
            safe_name = concept.replace(" ", "_")
            path = os.path.join("output", f"{safe_name}_d{depth}.png")
            sn.draw(concept, depth=depth, save_path=path)
            print()
        else:
            print("⚠ Opción no válida.\n")

if __name__ == "__main__":
    main()
