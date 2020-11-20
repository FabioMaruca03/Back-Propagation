package back.propagation.layers;

import java.util.ArrayList;
import java.util.List;

import back.propagation.neuron.H_units;

public class H_layers {

  private final int nU;
  private final List < H_units > hU;

  public H_layers(int nUnits) {
    this.nU = nUnits;
    hU = new ArrayList < >();
  }

  public void addHiddenUnit(H_units unit) {
    hU.add(unit);
  }

  public int getNumberOfUnits() {
    return nU;
  }

  public List < H_units > getHiddenUnits() {
    return hU;
  }

  @Override
  public String toString() {
    return "\n\nHL{" + "nU=" + nU + ", hU=" + hU + '}';
  }

}