package back.propagation.layers;

import java.util.ArrayList;
import java.util.List;

import back.propagation.neuron.O_Units;

public class O_layers {

  List<O_Units> hiddenUnits;

  public O_layers() {
    hiddenUnits = new ArrayList < >();
  }

  public void addOutputUnit(O_Units unit) {
    hiddenUnits.add(unit);
  }

  public List < O_Units > getOutputUnits() {
    return hiddenUnits;
  }

}