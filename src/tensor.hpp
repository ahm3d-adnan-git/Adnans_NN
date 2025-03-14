
#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <utility>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cassert>

inline constexpr size_t flatIdx(const std::vector< size_t >& shape, const std::vector< size_t >& idx)
{
    assert(shape.size() == idx.size());

    auto rank = idx.size();

    if (rank == 0)
    {
        return 0;
    }
    else if (rank == 1)
    {
        return idx[0];
    }
    else if (rank == 2)
    {
        return idx[0] * shape[1] + idx[1];
    }
    else
    {
        size_t flatIdx = 0;
        for (size_t i = 0; i < rank; i++)
        {
            size_t dimProduct = 1;
            for (size_t ii = i + 1; ii < rank; ii++)
            {
                dimProduct *= shape[ii];
            }
            flatIdx += idx[i] * dimProduct;
        }

        return flatIdx;
    }
}

inline size_t numTensorElements(const std::vector< size_t >& shape)
{
    size_t size = 1;
    for (auto d : shape)
    {
        size *= d;
    }
    return size;
}

template< typename ScalarType >
ScalarType stringToScalar(const std::string& str)
{
    std::stringstream s(str);
    ScalarType scalar;
    s >> scalar;
    return scalar;
}


template< class T >
concept Arithmetic = std::is_arithmetic_v< T >;

template< Arithmetic ComponentType >
class Tensor
{
public:
    // Constructs a tensor with rank = 0 and zero-initializes the element.
    Tensor();

    // Constructs a tensor with arbitrary shape and zero-initializes all elements.
    Tensor(const std::vector< size_t >& shape);

    // Constructs a tensor with arbitrary shape and fills it with the specified value.
    explicit Tensor(const std::vector< size_t >& shape, const ComponentType& fillValue);

    // Constructs a tensor with arbitrary shape and fills it with the specified vector of values.
    explicit Tensor(const std::vector< size_t >& shape, const std::vector< ComponentType >& values);

    // Copy-constructor.
    Tensor(const Tensor< ComponentType >& other);

    // Move-constructor.
    Tensor(Tensor< ComponentType >&& other) noexcept;

    // Copy-assignment
    Tensor&
    operator=(const Tensor< ComponentType >& other);

    // Move-assignment
    Tensor&
    operator=(Tensor< ComponentType >&& other) noexcept;

    // Destructor
    ~Tensor() = default;

    // Returns the rank of the tensor.
    [[nodiscard]] size_t rank() const;

    // Returns the shape of the tensor.
    [[nodiscard]] std::vector< size_t > shape() const;

    // Returns the number of elements of this tensor.
    [[nodiscard]] size_t numElements() const;

    // Element access function
    const ComponentType&
    operator()(const std::vector< size_t >& idx) const;

    // Element mutation function
    ComponentType&
    operator()(const std::vector< size_t >& idx);

    // xavier initialization
    Tensor<ComponentType> xavier_init();
    // Element-wise Addition
    Tensor<ComponentType> operator+(const Tensor<ComponentType>& other) const;
    // Element-wise Subtraction
    Tensor<ComponentType> operator-(const Tensor<ComponentType>& other) const;
    // Element-wise Multiplication
    Tensor<ComponentType> operator*(const Tensor<ComponentType>& other) const;
    // Element-wise Division
    Tensor<ComponentType> operator/(const Tensor<ComponentType>& other) const;
    // Scalar Subtraction
    Tensor<ComponentType> operator-(const ComponentType& scalar) const;
    // Scalar Multiplication
    Tensor<ComponentType> operator*(const ComponentType& scalar) const;
    // Scalar Division
    Tensor<ComponentType> operator/(const ComponentType& scalar) const;
    // Dot product
    Tensor<ComponentType> dot(const Tensor<ComponentType>& other) const;
    // Transpose
    Tensor<ComponentType> T() const;
    // sigmoid function
    Tensor<ComponentType> relu() const;
    // softmax function
    Tensor<ComponentType> softmax() const;
    // sigmoid derivative
    Tensor<ComponentType> relu_derivative() const;
    // cross entropy loss
    Tensor<ComponentType> cross_entropy_loss(const Tensor<ComponentType>& y) const;
    // slice
    Tensor<ComponentType> slice(size_t start, size_t end) const;
    // sum all
    double sum() const;
    // sum over axis
    Tensor<ComponentType> sum(int axis) const;

    // Reshape the tensor
    Tensor<ComponentType> reshape(const std::vector<size_t>& new_shape);

    // Repeat tensor along axis
    Tensor<ComponentType> repeat(size_t axis, size_t repeats) const;

    // max element
    Tensor<ComponentType> argmax(int axis) const;

    // Clip tensor values to a specified range
    Tensor<ComponentType> clip(ComponentType min_value, ComponentType max_value) const;

    // Logarithm of tensor elements
    Tensor<ComponentType> log() const;

    // He initialization
    Tensor<ComponentType> he_init() const;

private:

    std::vector< size_t > shape_;
    std::vector< ComponentType > data_;

};

template <Arithmetic ComponentType>
Tensor<ComponentType> Tensor<ComponentType>::he_init() const {
    Tensor<ComponentType> result(shape_);

    // Get the number of input connections (fan_in)
    size_t fan_in = shape_[1];  // Assuming shape is (output_size, input_size)

    // Standard deviation for He initialization
    double std_dev = std::sqrt(2.0 / fan_in);

    // Initialize weights with random values from N(0, std_dev)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<ComponentType> dist(0.0, std_dev);

    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = dist(gen);
    }

    return result;
}

template <Arithmetic ComponentType>
Tensor<ComponentType> Tensor<ComponentType>::log() const {
    Tensor<ComponentType> result(shape_); // Create a new tensor with the same shape
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::log(data_[i]); // Apply natural log element-wise
    }
    return result;
}

template<Arithmetic ComponentType>
Tensor<ComponentType> Tensor<ComponentType>::clip(ComponentType min_value, ComponentType max_value) const {
    Tensor<ComponentType> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::max(min_value, std::min(data_[i], max_value));
    }
    return result;
}

void print_shape(const std::vector< size_t >& shape)
{
    std::cout << "(";
    for (size_t i = 0; i < shape.size() - 1; i++)
    {
        std::cout << shape[i] << ", ";
    }
    std::cout << shape[shape.size() - 1] << ")\n";
}

// Broadcast `Tensor` to a new shape by repeating the elements  along the specified axis `repeats` times.
template <Arithmetic ComponentType>
Tensor<ComponentType> Tensor<ComponentType>::repeat(size_t repeat_rows, size_t repeat_cols) const {
    // Ensure tensor has at least 2 dimensions
    assert(rank() == 2 && "Repeat function currently supports only 2D tensors");

    // Get current shape
    size_t rows = shape_[0];
    size_t cols = shape_[1];

    // Define new shape
    size_t new_rows = rows * repeat_rows;
    size_t new_cols = cols * repeat_cols;
    Tensor<ComponentType> result({new_rows, new_cols});

    // Repeat the elements
    for (size_t i = 0; i < new_rows; ++i) {
        for (size_t j = 0; j < new_cols; ++j) {
            result({i, j}) = (*this)({i % rows, j % cols});
        }
    }

    return result;
}




// max element
template <Arithmetic ComponentType>
Tensor< ComponentType > Tensor<ComponentType>::argmax(int axis) const {
    assert(axis >= 0 && axis < static_cast<int>(shape_.size()) && "Invalid axis for argmax");

    std::vector<size_t> result_shape = {shape_[axis] == shape_[0] ? shape_[1] : shape_[0]};
    Tensor<double> result(result_shape);

    if (axis == 0) {  // Argmax along axis 0: Find max per column
        for (size_t j = 0; j < shape_[1]; ++j) {
            size_t max_index = 0;
            ComponentType max_value = (*this)({0, j});
            for (size_t i = 1; i < shape_[0]; ++i) {
                if ((*this)({i, j}) > max_value) {
                    max_value = (*this)({i, j});
                    max_index = i;
                }
            }
            result({j}) = max_index;  // Store in 1D result tensor
        }
    } 
    else if (axis == 1) {  // Argmax along axis 1: Find max per row
        for (size_t i = 0; i < shape_[0]; ++i) {
            size_t max_index = 0;
            ComponentType max_value = (*this)({i, 0});
            for (size_t j = 1; j < shape_[1]; ++j) {
                if ((*this)({i, j}) > max_value) {
                    max_value = (*this)({i, j});
                    max_index = j;
                }
            }
            result({i}) = max_index;  // Store in 1D result tensor
        }
    } 
    else {
        throw std::invalid_argument("argmax currently supports only 2D tensors");
    }
    
    return result;
}


template <Arithmetic ComponentType>
Tensor< ComponentType > Tensor< ComponentType >::reshape(const std::vector<size_t>& new_shape) {
    // Compute the total number of elements in both shapes
    size_t new_num_elements = numTensorElements(new_shape);
    size_t old_num_elements = numTensorElements(shape_);
    // std::cerr << "new_num_elements: " << new_num_elements << std::endl;
    // std::cerr << "old_num_elements: " << old_num_elements << std::endl;
    // std::cerr << "new_shape: " << std::endl;
    // print_shape(new_shape);
    // std::cerr << "old_shape: " << std::endl;
    // print_shape(shape_);
    // Ensure reshaping is valid
    assert(new_num_elements == old_num_elements && "Invalid reshape! Total elements must remain the same.");

    // Create a new tensor with the new shape but keep the same data
    Tensor<double> reshaped_tensor(new_shape, data_);

    return reshaped_tensor; 
}

// Subtract scalar
template< Arithmetic ComponentType >
Tensor< ComponentType > Tensor< ComponentType >::operator-(const ComponentType& scalar) const
{
    Tensor< ComponentType > result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - scalar;
    }
    return result;
}

// Sum all
template< Arithmetic ComponentType >
double Tensor< ComponentType >::sum() const
{
    double result = 0;
    for (size_t i = 0; i < data_.size(); ++i) {
        result += data_[i];
    }
    return result;
}

// Sum over axis
template< Arithmetic ComponentType >
Tensor< ComponentType > Tensor< ComponentType >::sum(int axis) const
{
    if (axis == 1) { 
        Tensor< ComponentType > result({shape_[0], 1});
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result({i, 0}) += (*this)({i, j});
            }
        }
        return result;
    } else if (axis == 0) {
        Tensor< ComponentType > result({1, shape_[1]});
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result({0, j}) += (*this)({i, j});                
            }
        }
        return result;
    } else {
        throw std::invalid_argument("Invalid axis");
    }
}

// Slice
template< Arithmetic ComponentType >
Tensor< ComponentType > Tensor< ComponentType >::slice(size_t start, size_t end) const
{
    assert(rank() == 2 && "Only 2D tensors can be sliced!");

    size_t num_rows = end - start;
    Tensor< ComponentType > result({num_rows, shape_[1]});

    for (size_t i = 0; i < num_rows; ++i)
    {
        for (size_t j = 0; j < shape_[1]; ++j)
        {
            result({i, j}) = (*this)({start + i, j});
        }
    }

    return result;
}

// Extra constructor
template< Arithmetic ComponentType >
Tensor< ComponentType >::Tensor(const std::vector< size_t >& shape, const std::vector< ComponentType >& values)
    : shape_(shape), data_(values)
{
}

//xavier initialization
template<Arithmetic ComponentType>
Tensor<ComponentType> Tensor<ComponentType>::xavier_init() {
    std::random_device rd;
    std::mt19937 gen(rd());

    double stddev = std::sqrt(2.0 / (shape_[0] + shape_[1])); // Correct Xavier formula
    std::normal_distribution<ComponentType> dist(0.0, stddev); // Use ComponentType instead of T

    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] = dist(gen);
    }

    return *this;  // Ensure the function returns the modified Tensor
}

// Element-wise Addition
template<Arithmetic ComponentType>
Tensor<ComponentType> Tensor<ComponentType>::operator+(const Tensor<ComponentType>& other) const {
    // std::cerr << "shape_: " << shape_[0] << " " << shape_[1] << std::endl;
    // std::cerr << "other.shape_: " << other.shape_[0] << " " << other.shape_[1] << std::endl;
    assert(shape_ == other.shape_ && "Tensors must have the same shape for addition!");

    Tensor<ComponentType> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

// Element-wise Subtraction
template< Arithmetic ComponentType >
Tensor<ComponentType> Tensor<ComponentType>::operator-(const Tensor<ComponentType>& other) const {
    // std::cerr << "subtraction  " << std::endl;
    // std::cerr << "shape_: " << shape_[0] << " " << shape_[1] << std::endl;
    // std::cerr << "other.shape_: " << other.shape_[0] << " " << other.shape_[1] << std::endl;
    assert(shape_ == other.shape_ && "Tensors must have the same shape for subtraction!");
    Tensor<ComponentType> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

// Element-wise Multiplication
template< Arithmetic ComponentType >
Tensor<ComponentType> Tensor<ComponentType>::operator*(const Tensor<ComponentType>& other) const {
    assert(shape_ == other.shape_ && "Tensors must have the same shape for multiplication!");

    Tensor<ComponentType> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

// Element-wise Division
template< Arithmetic ComponentType >
Tensor<ComponentType> Tensor<ComponentType>::operator/(const Tensor<ComponentType>& other) const {
    assert(shape_ == other.shape_ && "Tensors must have the same shape for division!");

    Tensor<ComponentType> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] / other.data_[i];
    }
    return result;
}

// Scalar Multiplication
template< Arithmetic ComponentType >
Tensor<ComponentType> Tensor<ComponentType>::operator*(const ComponentType& scalar) const {
    Tensor<ComponentType> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

//Scalar Division
template< Arithmetic ComponentType >
Tensor<ComponentType> Tensor<ComponentType>::operator/(const ComponentType& scalar) const {
    Tensor<ComponentType> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] / scalar;
    }
    return result;
}

// Dot product
template< Arithmetic ComponentType >
Tensor<ComponentType> Tensor<ComponentType>::dot(const Tensor<ComponentType>& other) const {

    assert(shape_[1] == other.shape_[0] && "Tensors must have the same shape for dot product!");

    size_t M = shape_[0];  // Rows of A
    size_t N = other.shape_[1];  // Cols of B
    size_t K = shape_[1];  // Common dimension

    Tensor<ComponentType> result({M, N});

    // O(n²) optimization: Process row-wise
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            ComponentType val = data_[i * K + k];  // Cache value
            for (size_t j = 0; j < N; ++j) {
                result.data_[i * N + j] += val * other.data_[k * N + j];
            }
        }
    }
    return result;
}

// Transpose
template< Arithmetic ComponentType >
Tensor<ComponentType> Tensor<ComponentType>::T() const {
    assert(rank() == 2 && "Only 2D tensors can be transposed!");

    Tensor<ComponentType> result({shape_[1], shape_[0]});
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            result({j, i}) = (*this)({i, j});
        }
    }
    return result;
}

// sigmoid function
template< Arithmetic ComponentType >
Tensor<ComponentType> Tensor<ComponentType>::relu() const {
    Tensor<ComponentType> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::max(data_[i], 0.01 * data_[i]); // LeakyReLU with alpha = 0.01
    }
    return result;
}

// softmax function
template<Arithmetic ComponentType>
Tensor<ComponentType> Tensor<ComponentType>::softmax() const {
    Tensor<ComponentType> result(shape_);  // Initialize result tensor

    // Find the max value to improve numerical stability
    ComponentType max_value = *std::max_element(data_.begin(), data_.end());

    // Compute exp(data_[i] - max_value) for each element and sum up the values
    ComponentType sum_exp = 0;
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::exp(data_[i] - max_value);  // Apply exp shift
        sum_exp += result.data_[i];  // Accumulate the sum
    }

    // Normalize each value
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] /= sum_exp;
    }

    return result;
}


// sigmoid derivative
template< Arithmetic ComponentType >
Tensor<ComponentType> Tensor<ComponentType>::relu_derivative() const {
    Tensor<ComponentType> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] > 0 ? 1 : 0;
    }
    return result;
}

// cross entropy loss
template< Arithmetic ComponentType >
Tensor<ComponentType> Tensor<ComponentType>::cross_entropy_loss(const Tensor<ComponentType>& y) const {
    
    // // std::cerr << "cross_entropy_loss" << std::endl;
    // // std::cerr << "shape Y_pred: " << std::endl;
    // print_shape(shape_);
    // // std::cerr << "shape Y_true: " << std::endl;
    // print_shape(y.shape_);
    // // std::cerr << "before assert " << std::endl;
    assert(shape_ == y.shape_ && "Tensors must have the same shape for cross entropy loss!");
    // // std::cerr << "after assert " << std::endl;
    Tensor<ComponentType> result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = -y.data_[i] * std::log(data_[i]);
    }
    // // std::cerr << "after cross_entropy_loss calculation" << std::endl;
    return result;
}

template< Arithmetic ComponentType >
Tensor< ComponentType >::Tensor()
    : shape_(0), data_(1, 0)
{
}

template< Arithmetic ComponentType >
Tensor< ComponentType >::Tensor(const std::vector< size_t >& shape)
    : shape_(shape), data_(numTensorElements(shape), 0)
{
}

template< Arithmetic ComponentType >
Tensor< ComponentType >::Tensor(const std::vector< size_t >& shape, const ComponentType& fillValue)
    : shape_(shape), data_(numTensorElements(shape), fillValue)
{
}

// Copy-assignment
template< Arithmetic ComponentType >
Tensor< ComponentType >::Tensor(const Tensor< ComponentType >& other) = default;


// Move-constructor.
template< Arithmetic ComponentType >
Tensor< ComponentType >::Tensor(Tensor< ComponentType >&& other) noexcept
    : shape_(std::exchange(other.shape_, std::vector< size_t >())), data_(std::exchange(other.data_, {0}))
{
}

// Copy-assignment
template< Arithmetic ComponentType >
Tensor< ComponentType >& Tensor< ComponentType >::operator=(const Tensor< ComponentType >& other) = default;


// Move-assignment
template< Arithmetic ComponentType >
Tensor< ComponentType >& Tensor< ComponentType >::operator=(Tensor< ComponentType >&& other) noexcept

{
    shape_ = std::exchange(other.shape_, std::vector< size_t >());
    data_ = std::exchange(other.data_, {0});
    return *this;
}

template< Arithmetic ComponentType >
size_t
Tensor< ComponentType >::rank() const
{
    return shape_.size();
}

template< Arithmetic ComponentType >
std::vector< size_t >
Tensor< ComponentType >::shape() const
{
    return shape_;
}

template< Arithmetic ComponentType >
size_t
Tensor< ComponentType >::numElements() const
{
    return numTensorElements(shape_);
}

template< Arithmetic ComponentType >
const ComponentType&
Tensor< ComponentType >::operator()(const std::vector< size_t >& idx) const
{
    assert(idx.size() == rank());
    return data_[flatIdx(shape_, idx)];
}

template< Arithmetic ComponentType >
ComponentType&
Tensor< ComponentType >::operator()(const std::vector< size_t >& idx)
{
    assert(idx.size() == rank());
    return data_[flatIdx(shape_, idx)];
}


// Returns true if the shapes and all elements of both tensors are equal.
template< Arithmetic ComponentType >
bool operator==(const Tensor< ComponentType >& a, const Tensor< ComponentType >& b)
{

    if (a.shape() != b.shape())
    {
        return false;
    }

    size_t rank = a.rank();
    std::vector< size_t > shape = a.shape();
    size_t numElements = a.numElements();

    bool equal = true;

    if (rank == 0)
    {
        std::vector< size_t > idx(0);
        return a(idx) == b(idx);
    }
    else if (rank == 1)
    {
        for (size_t i = 0; i < shape[0]; i++)
        {
            std::vector< size_t > idx(1);
            idx[0] = i;
            equal &= a(idx) == b(idx);
        }
    }
    else
    {
        size_t cnt = 0;
        std::vector< size_t > idx(rank, 0);

        while (cnt < numElements)
        {
            for (size_t i = 0; i < shape[rank - 1]; i++)
            {
                equal &= a(idx) == b(idx);
                idx[rank - 1]++;
            }

            idx[rank - 1]++;

            for (size_t i = rank - 1; i > 0; i--)
            {
                if (idx[i] >= shape[i])
                {
                    idx[i] = 0;
                    idx[i - 1]++;
                }
            }

            cnt += shape[rank - 1];
        }
    }

    return equal;
}

// Pretty-prints the tensor to stdout.
// This is not necessary (and not covered by the tests) but nice to have, also for debugging (and for exercise of course...).
template< Arithmetic ComponentType >
std::ostream&
operator<<(std::ostream& out, const Tensor< ComponentType >& tensor)
{

    if (tensor.rank() == 0)
    {
        std::vector< size_t > idx(0);
        out << "() [" << tensor(idx) << "]\n";
    }
    else if (tensor.rank() == 1)
    {
        out << "(:) [";
        for (size_t i = 0; i < tensor.shape()[0] - 1; i++)
        {
            std::vector< size_t > idx(1);
            idx[0] = i;
            out << tensor(idx) << " ";
        }
        std::vector< size_t > idx(1);
        idx[0] = tensor.shape()[0] - 1;
        out << tensor(idx) << "]\n";
    }
    else
    {
        size_t cnt = 0;
        std::vector< size_t > idx(tensor.rank(), 0);

        while (cnt < tensor.numElements())
        {
            out << "(";
            for (size_t i = 0; i < tensor.rank() - 1; i++)
            {
                out << idx[i] << ", ";
            }
            out << ":) [";
            for (size_t i = 0; i < tensor.shape()[tensor.rank() - 1] - 1; i++)
            {
                out << tensor(idx) << " ";
                idx[tensor.rank() - 1]++;
            }

            out << tensor(idx) << "]\n";
            idx[tensor.rank() - 1]++;

            for (size_t i = tensor.rank() - 1; i > 0; i--)
            {
                if (idx[i] >= tensor.shape()[i])
                {
                    idx[i] = 0;
                    idx[i - 1]++;
                }
            }

            cnt += tensor.shape()[tensor.rank() - 1];
        }
    }

    return out;
}

// Reads a tensor from file.
template< Arithmetic ComponentType >
Tensor< ComponentType > readTensorFromFile(const std::string& filename)
{

    std::ifstream file;
    file.open(filename);

    if (!file.is_open())
    {
        std::cerr << "Could not open file." << std::endl;
        std::exit(1);
    }

    std::string line;
    std::getline(file, line);

    auto rank = stringToScalar< size_t >(line);

    std::vector< size_t > shape(rank);
    for (size_t i = 0; i < rank; i++)
    {
        std::getline(file, line);
        shape[i] = stringToScalar< size_t >(line);
    }

    Tensor< ComponentType > tensor(shape);

    if (rank == 0)
    {
        std::getline(file, line);
        tensor(shape) = stringToScalar< ComponentType >(line);
    }
    else
    {
        std::vector< size_t > idx(shape.size(), 0);
        size_t cnt = 0;
        while (cnt < tensor.numElements())
        {
            std::getline(file, line);
            tensor(idx) = stringToScalar< ComponentType >(line);

            idx[rank - 1]++;
            for (size_t i = rank - 1; i > 0; i--)
            {
                if (idx[i] >= shape[i])
                {
                    idx[i] = 0;
                    idx[i - 1]++;
                }
            }

            cnt++;
        }
    }

    file.close();
    return tensor;
}

// Writes a tensor to file.
template< Arithmetic ComponentType >
void writeTensorToFile(const Tensor< ComponentType >& tensor, const std::string& filename)
{

    std::ofstream file;
    file.open(filename);

    file << tensor.rank() << "\n";
    for (auto d : tensor.shape())
    {
        file << d << "\n";
    }

    if (tensor.rank() == 0)
    {
        file << tensor({}) << "\n";
    }
    else
    {
        std::vector< size_t > idx(tensor.shape().size(), 0);
        size_t cnt = 0;
        while (cnt < tensor.numElements())
        {
            file << tensor(idx) << "\n";

            idx[tensor.rank() - 1]++;
            for (size_t i = tensor.rank() - 1; i > 0; i--)
            {
                if (idx[i] >= tensor.shape()[i])
                {
                    idx[i] = 0;
                    idx[i - 1]++;
                }
            }

            cnt++;
        }
    }

    file.close();
}

