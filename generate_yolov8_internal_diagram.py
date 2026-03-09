from graphviz import Digraph
import os


def verify_graphviz():
    import shutil
    graphviz_bin = shutil.which('dot')
    if graphviz_bin:
        return True
    if os.path.exists(r"C:\Program Files\Graphviz\bin\dot.exe"):
        os.environ['PATH'] = r"C:\Program Files\Graphviz\bin;" + os.environ.get('PATH', '')
        return True
    print("ERROR: Graphviz 'dot' not found. Please add Graphviz to PATH or install it.")
    return False


def create_yolov8_internal_diagram():
    """Generates a diagram that explains the internal architecture of YOLOv8.

    The diagram is a simplified conceptual view (not a literal layer-by-layer
    dump). It highlights common modules used in modern YOLOv8-like networks:
    - Input / preprocessing
    - Stem / focus block
    - Backbone (C2f / CSP / conv stacks + SPPF)
    - Neck (FPN / PAN-style feature aggregation)
    - Head (detection/classification heads)
    - Postprocessing (NMS, confidence thresholding)
    """

    dot = Digraph('YOLOv8_Internal', comment='YOLOv8 Internal Architecture')
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box', style='rounded')

    dot.node('Input', 'Input Image\n(Resized, Normalized)')
    dot.node('Stem', 'Stem / Focus\n(initial convs, patch merging)')

    with dot.subgraph(name='cluster_backbone') as c:
        c.attr(label='Backbone')
        c.node('ConvStack', 'Conv / Bottleneck stacks')
        c.node('C2f', 'C2f / CSP Modules\n(gradient flow & efficient feature reuse)')
        c.node('SPPF', 'SPPF (Spatial Pyramid Pooling - Fast)\n(multi-scale receptive fields)')

    with dot.subgraph(name='cluster_neck') as c:
        c.attr(label='Neck')
        c.node('FPN', 'PANet / FPN-like \n(feature aggregation across scales)')

    with dot.subgraph(name='cluster_head') as c:
        c.attr(label='Head')
        c.node('DetectHead', 'Detection Head\n(class logits + bbox preds / anchors or anchor-free)')
        c.node('ClsHead', 'Classification Head (if used)')

    dot.node('Post', 'Postprocessing\n(Confidence thresholding, NMS)')
    dot.node('Output', 'Outputs:\nBoxes, Classes, Confidences\n(or class probabilities for cls variant)')


    dot.edge('Input', 'Stem')
    dot.edge('Stem', 'ConvStack')
    dot.edge('ConvStack', 'C2f')
    dot.edge('C2f', 'SPPF')
    dot.edge('SPPF', 'FPN')
    dot.edge('FPN', 'DetectHead')
    dot.edge('FPN', 'ClsHead')
    dot.edge('DetectHead', 'Post')
    dot.edge('ClsHead', 'Post')
    dot.edge('Post', 'Output')

    dot.node('backbone_caption', 'Backbone: extracts hierarchical features\n(from low-level edges to high-level semantics)', shape='note', fontsize='10')
    dot.node('neck_caption', 'Neck: fuses multi-scale features\nso heads see both local and context information', shape='note', fontsize='10')
    dot.node('head_caption', 'Head: predicts objectness / classes / boxes\nfor each spatial location or proposal', shape='note', fontsize='10')

    dot.edge('C2f', 'backbone_caption', style='dashed', arrowhead='none')
    dot.edge('FPN', 'neck_caption', style='dashed', arrowhead='none')
    dot.edge('DetectHead', 'head_caption', style='dashed', arrowhead='none')

    dot.node('legend', 'C2f: Concatenation-based efficient block\nSPPF: faster multi-scale context pooling', shape='note', fontsize='9')
    dot.edge('SPPF', 'legend', style='dashed', arrowhead='none')

    out_dir = os.path.join(os.getcwd(), 'architecture')
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, 'yolov8_internal.png')
    svg_path = os.path.join(out_dir, 'yolov8_internal.svg')

    dot.render(os.path.splitext(png_path)[0], format='png', cleanup=True)
    dot.render(os.path.splitext(svg_path)[0], format='svg', cleanup=True)

    print(f"YOLOv8 internal architecture diagrams generated:\n  {png_path}\n  {svg_path}")


if __name__ == '__main__':
    if not verify_graphviz():
        raise SystemExit(1)
    create_yolov8_internal_diagram()
