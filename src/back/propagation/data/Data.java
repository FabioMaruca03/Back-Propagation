package back.propagation.data;

import java.util.ArrayList;
import java.util.List;

public class Data {

  private List<List<Double>> x;
  private List<Double> y;

  private List<Double> maxX;
  private Double maxY;
  private List<Double> minX;
  private Double minY  ;

  public Data(List < List < Double >> x, List < Double > y) {
    this.x = x;
    this.y = y;

    init();
    normalize();
  }

  private void normalize() {

    if (!x.isEmpty()) {
      for (List<Double> doubles : x) {
        for (int j = 0; j < minX.size(); ++j) {
          double normalized = (doubles.get(j) - minX.get(j)) / (maxX.get(j) - minX.get(j));
          doubles.set(j, normalized);
        }
      }
    }

    if (!y.isEmpty()) {
      for (int i = 0; i < y.size(); ++i) {
        double normalized = (y.get(i) - minY) / (maxY - minY);
        y.set(i, normalized);
      }
    }

  }

  private void init() {
    this.maxX = new ArrayList<>();
    this.minX = new ArrayList<>();

    if (!x.isEmpty()) {
      for (int i = 0; i < x.get(0).size(); ++i) {
        double temp = 0.0;
        double tempMin = Double.MAX_VALUE;

        for (List<Double> doubles : x) {
          if (doubles.get(i) > temp) {
            temp = doubles.get(i);
          }
          if (doubles.get(i) < tempMin) {
            tempMin = doubles.get(i);
          }
        }
        maxX.add(temp);
        minX.add(tempMin);
      }
    }

    if (!y.isEmpty()) {
      double temp = 0.0;
      double tempMin = Double.MAX_VALUE;

      for (Double aDouble : y) {
        if (aDouble > temp) {
          temp = aDouble;
        }

        if (aDouble < tempMin) {
          tempMin = aDouble;
        }
      }
      maxY = temp;
      minY = tempMin;
    }
  }

  public List<List<Double>> getX() {
    return x;
  }

  public void setX(List<List<Double>> x) {
    this.x = x;
  }

  public List<Double> getY() {
    return y;
  }

  public void setY(List<Double> y) {
    this.y = y;
  }

  public List<Double> getMaxX() {
    return maxX;
  }

  public void setMaxX(List<Double> maxX) {
    this.maxX = maxX;
  }

  public Double getMaxY() {
    return maxY;
  }

  public void setMaxY(Double maxY) {
    this.maxY = maxY;
  }

  public List < Double > getMinX() {
    return minX;
  }

  public void setMinX(List<Double> minX) {
    this.minX = minX;
  }

  public Double getMinY() {
    return minY;
  }

  public void setMinY(Double minY) {
    this.minY = minY;
  }

}