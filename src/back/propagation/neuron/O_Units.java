package back.propagation.neuron;

public class O_Units extends N_units {

  private double eV;

  public O_Units(int nPreviousNodes) {
    super(nPreviousNodes);
  }

  public double geteV() {
    return eV;
  }

  public void seteV(double eV) {
    this.eV = eV;
  }

}