#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

// number of rows, columns and steps
#define M 10
#define N 10
#define T 10

// initialize the grid with randomized 0 or 1's
void initialize(int grid[M][N])
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            grid[i][j] = rand() % 2;
        }
    }
}

// print the grid
void printGrid(int grid[M][N])
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", grid[i][j]);
        }
        // finished printing a row, move to the next row
        printf("\n");
    }
    printf("\n");
}

// calculate live neighbors of a cell
int calculateLiveNeighbors(int grid[M][N], int x, int y)
{
    int liveNeighbors = 0;
    // check the 8 neighbors of the cell (up, down, left, right, and diagonals)
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int newX = x + i;
            int newY = y + j;
            if (newX >= 0 && newX < M && newY >= 0 && newY < N)
            {
                liveNeighbors += grid[newX][newY];
            }
        }
    }
    // subtract cell itself (don't include the cell itself in the amount of live neighbors)
    return liveNeighbors - grid[x][y];
}

int main(int argc, char *argv[])
{
    int rank, size;
    int grid[M][N];
    int nextGrid[M][N];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // seed the random number generator with a different seed for each process
    srand(time(NULL) + rank);

    int localM = M / size;
    int localGrid[localM + 2][N];
    int localNextGrid[localM + 2][N];

    if (rank == 0)
    {
        initialize(grid);
    }

    MPI_Scatter(grid, localM * N, MPI_INT, &localGrid[1], localM * N, MPI_INT, 0, MPI_COMM_WORLD);

    for (int t = 0; t < T; t++)
    {
        // exchange data held in adjacent processes
        if (rank != 0)
        {
            MPI_Send(localGrid[1], N, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(localGrid[0], N, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1)
        {
            MPI_Send(localGrid[localM], N, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(localGrid[localM + 1], N, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int i = 1; i <= localM; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int liveNeighbors = calculateLiveNeighbors(localGrid, i, j);
                if (localGrid[i][j] == 1)
                {
                    // if cell is alive check if it has less than 2 or more than 3 live neighbors and update its state accordingly
                    localNextGrid[i][j] = (liveNeighbors < 2 || liveNeighbors > 3) ? 0 : 1;
                }
                else
                {
                    // if cell is dead check if it has 3 neighbors and update its state accordingly
                    localNextGrid[i][j] = (liveNeighbors == 3) ? 1 : 0;
                }
            }
        }
        // copy the next grid to the current grid
        for (int i = 1; i <= localM; i++)
        {
            for (int j = 0; j < N; j++)
            {
                localGrid[i][j] = localNextGrid[i][j];
            }
        }

        // Optional printing of the local grids of each process.
        // Uncomment lines (125-138) to print the local grids of each process at each time step.
        // // print the local grid of each process
        // printf("Rank %d, Time Step %d:\n", rank, t);
        // for (int i = 1; i <= localM; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //     {
        //         printf("%d ", localGrid[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");

        // // Ensure all processes have finished printing
        // MPI_Barrier(MPI_COMM_WORLD);
    }

    // collect the local grids from all processes to the global grid
    MPI_Gather(&localGrid[1], localM * N, MPI_INT, grid, localM * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Final Grid:\n");
        printGrid(grid);
    }

    MPI_Finalize();
    return 0;
}
