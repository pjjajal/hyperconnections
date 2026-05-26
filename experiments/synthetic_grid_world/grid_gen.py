import torch


def generate_grid(n_grids, n_rows, n_cols, colours, generator: torch.Generator):
    """Generate a grid world dataset.

    Args:
        n_grids: Number of grids to generate
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        colours: List of possible colours for each cell
        generator: Torch random number generator

    Returns:
        A tensor of shape [n_grids, n_rows, n_cols] with colour indices
    """
    if generator is not None:
        torch.manual_seed(generator)

    n_colours = len(colours)
    positions = torch.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    grid = torch.randint(0, n_colours, (n_grids, n_rows, n_cols), generator=generator)
    return grid, positions




class GridWorldDataset(torch.utils.data.Dataset):
    """
    Dataset of synthetic grid worlds. Each grid is a 2D array of colour indices.
    """

    def __init__(self, n_grids, n_rows, n_cols, colours, seed: int = 42):
        generator = torch.Generator().manual_seed(seed)
        self.grids, self.positions = generate_grid(n_grids, n_rows, n_cols, colours, generator)

        


    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        return self.grids[idx], self.positions


