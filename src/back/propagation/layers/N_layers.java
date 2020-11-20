package back.propagation.layers;

import java.util.ArrayList;
import java.util.List;

import back.propagation.Node;

public class N_layers {

  List<Node> hU;

  public N_layers() {
    hU = new ArrayList < >();
  }

  public void addInputNode(Node node) {
    hU.add(node);
  }

  public List<Node> getInputNodes() {
    return hU;
  }

}