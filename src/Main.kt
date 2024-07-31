import com.ustermetrics.ecos4j.Model
import com.ustermetrics.ecos4j.Parameters
import com.ustermetrics.ecos4j.Status.OPTIMAL
import org.ejml.data.DMatrixSparseCSC
import org.ejml.dense.row.factory.DecompositionFactory_DDRM
import org.ejml.ops.DConvertMatrixStruct
import org.ejml.simple.SimpleMatrix

fun toLongArray(arr: IntArray): LongArray {
    return arr.map { it.toLong() }.toLongArray()
}

fun main() {
    print(Model.version())

    // Define portfolio optimization problem
    val mu = SimpleMatrix(doubleArrayOf(0.05, 0.06, 0.08, 0.06))
    val sigma = SimpleMatrix(
        4, 4, true,
        0.0225, 0.003, 0.015, 0.0225,
        0.003, 0.04, 0.035, 0.024,
        0.015, 0.035, 0.0625, 0.06,
        0.0225, 0.024, 0.06, 0.09
    )
    val sigmaLimit = 0.2

    // Problem dimension
    val n = mu.getNumRows()

    // Compute Cholesky decomposition of sigma
    val chol = DecompositionFactory_DDRM.chol(n, true)
    if (!chol.decompose(sigma.getMatrix()))
        throw IllegalStateException("Cholesky decomposition failed")
    val upTriMat = SimpleMatrix.wrap(chol.getT(null)).transpose()

    // Define second-order cone program
    val cMat = mu.negative()
        .concatRows(SimpleMatrix(1, 1))
    System.out.println("\ncMat")
    cMat.print()

    val aMat = SimpleMatrix.ones(1, n)
        .concatColumns(SimpleMatrix(1, 1))
    System.out.println("\naMat")
    aMat.print()

    val bMat = SimpleMatrix.ones(1, 1)
    System.out.println("\nbMat")
    bMat.print()

    val gMatPosOrt = SimpleMatrix.identity(n)
        .negative()
        .concatColumns(SimpleMatrix(n, 1))
        .concatRows(SimpleMatrix(1, n).concatColumns(SimpleMatrix.ones(1, 1)))
    val gMatSoc = SimpleMatrix(1, n)
        .concatColumns(SimpleMatrix.filled(1, 1, -1.0))
        .concatRows(upTriMat.negative().concatColumns(SimpleMatrix(n, 1)))
    val gMat = gMatPosOrt.concatRows(gMatSoc)
    System.out.println("\ngMat")
    gMat.print()

    val hMat = SimpleMatrix(2 * n + 2, 1)
    hMat.set(n, 0, sigmaLimit)
    System.out.println("\nhMat")
    hMat.print()

    // ecos4j needs sparse aMat and gMat
    val tol = 1e-8
    val aSpMat = DConvertMatrixStruct.convert(aMat.getDDRM(), null as DMatrixSparseCSC?, tol)
    System.out.println("\naSpMat")
    aSpMat.print()

    val gSpMat = DConvertMatrixStruct.convert(gMat.getDDRM(), null as DMatrixSparseCSC?, tol)
    System.out.println("\ngSpMat")
    gSpMat.print()

    Model().use {
        // Set up model
        it.setup(
            n + 1L, longArrayOf(n + 1L), 0, gSpMat.nz_values, toLongArray(gSpMat.col_idx),
            toLongArray(gSpMat.nz_rows), cMat.getDDRM().data, hMat.getDDRM().data, aSpMat.nz_values,
            toLongArray(aSpMat.col_idx), toLongArray(aSpMat.nz_rows), bMat.getDDRM().data
        )

        // Create and set parameters
        val parameters = Parameters.builder()
            .verbose(true)
            .build()
        it.setParameters(parameters)

        // Optimize model
        val status = it.optimize()
        if (status != OPTIMAL)
            throw IllegalStateException("Optimization failed")

        // Get solution
        val xMat = SimpleMatrix(it.x())
        System.out.println("xMat")
        xMat.print()
    }
}
