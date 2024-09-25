# mc_conf
Tool to compute confidence intervals through Monte-Carlo sampling of error-bars.


## Example Usage

```python
#        1 ± 0.5  ,  2 ± 0.3  ,  1 ± 0.3
data = [(1.0, 0.5), (2.0, 0.3), (1.0, 0.3)]

sample = np.linspace(-1.0, 1.0, 20)
func = lambda x, a, b, c: a*x**2 + b*x + c
best, *max_min = mc_confidence_interval(sample, func, params=data, n=50, calc_best=True)

plot_confidence_interval(sample, *max_min, best, file_name='my_plot.png')
```
