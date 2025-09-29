# Physical Loss Weights Guide

This guide explains how to use the physical constraint loss weights in the GNN TSS prediction training.

## Available Loss Weights

| Parameter | Description | Default | Range | Effect |
|-----------|-------------|---------|-------|--------|
| `--lambda_mono` | Monotonicity loss weight | 0.15 | 0.0 - 1.0 | Ensures activation variance relationships are maintained |
| `--lambda_shortcut` | Shortcut awareness loss weight | 0.001 | 0.0 - 0.5 | Highlights the effect of shortcuts in TSS prediction |
| `--lambda_consistency` | Consistency loss weight | 0.0 | 0.0 - 0.5 | Ensures TSS predictions are consistent with shortcut characteristics |
| `--lambda_ranking` | Ranking loss weight | 0.0 | 0.0 - 0.5 | Maintains proper ranking based on shortcut characteristics |

## Usage Examples

### Basic Training (Default Weights)
```bash
python main.py --architecture mobilenet
```

### High Monotonicity Constraint
```bash
python main.py --lambda_mono 0.5 --architecture mobilenet
```
- **Use when**: You want to strongly enforce monotonic relationships in activation variances
- **Effect**: Model will be more conservative in predicting TSS changes

### Strong Shortcut Awareness
```bash
python main.py --lambda_shortcut 0.1 --lambda_consistency 0.2 --architecture mobilenet
```
- **Use when**: You want the model to pay special attention to shortcut connections
- **Effect**: Model will better capture the impact of skip connections on TSS

### Balanced Physical Constraints
```bash
python main.py --lambda_mono 0.3 --lambda_shortcut 0.05 --lambda_consistency 0.1 --lambda_ranking 0.05
```
- **Use when**: You want a balanced approach with all constraints
- **Effect**: Model will respect all physical relationships while maintaining good performance

### High Physical Constraints (Research Mode)
```bash
python main.py --lambda_mono 0.8 --lambda_shortcut 0.2 --lambda_consistency 0.3 --lambda_ranking 0.2
```
- **Use when**: You want to strongly enforce physical constraints for research
- **Effect**: Model will prioritize physical correctness over raw accuracy

## Weight Selection Guidelines

### For Production Use
- Start with default weights
- Gradually increase `--lambda_mono` if monotonicity is important
- Add `--lambda_shortcut` if shortcut connections are critical

### For Research
- Experiment with higher weights to understand physical constraints
- Use `--lambda_consistency` and `--lambda_ranking` for advanced analysis
- Monitor both accuracy and physical constraint satisfaction

### For Debugging
- Set all weights to 0.0 to disable physical constraints
- Gradually add constraints one by one to see their individual effects
- Use very high weights to see maximum constraint enforcement

## Monitoring Physical Constraints

The training output will show the individual loss components:
```
Epoch 10 | Train Loss: 0.123456 | Val Loss: 0.098765 | Test RMSE: 0.045678 | Time: 2.34s
Loss Components: MSE: 0.100000, Mono: 0.015000, Shortcut: 0.005000, Consistency: 0.002000, Ranking: 0.001000
```

## Tips for Weight Tuning

1. **Start Small**: Begin with default weights and increase gradually
2. **Monitor Convergence**: Watch for training instability with high weights
3. **Balance Trade-offs**: Higher physical constraints may reduce raw accuracy
4. **Architecture Matters**: MobileNet v2 may need different weights than EfficientNet
5. **Data Dependent**: Adjust weights based on your specific dataset characteristics

## Common Weight Combinations

### Conservative (High Accuracy)
```bash
python main.py --lambda_mono 0.05 --lambda_shortcut 0.001
```

### Balanced (Good Accuracy + Physics)
```bash
python main.py --lambda_mono 0.2 --lambda_shortcut 0.01 --lambda_consistency 0.05
```

### Physics-First (Research)
```bash
python main.py --lambda_mono 0.5 --lambda_shortcut 0.1 --lambda_consistency 0.2 --lambda_ranking 0.1
```

### Debugging (No Constraints)
```bash
python main.py --lambda_mono 0.0 --lambda_shortcut 0.0 --lambda_consistency 0.0 --lambda_ranking 0.0
```




