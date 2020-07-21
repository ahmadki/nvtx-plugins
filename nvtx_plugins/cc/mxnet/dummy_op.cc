#include <mxnet/lib_api.h>
#include <mxnet/base.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>

namespace mxnet {
namespace op {

inline bool shape_assign(mxnet::TShape *y, const mxnet::TShape& x) {
  if (!mxnet::ndim_is_known(*y)) {
    *y = x;
    return true;
  } else if (y->ndim() != x.ndim()) {
    return !mxnet::ndim_is_known(x);
  } else {
    for (int i = 0; i < y->ndim(); ++i) {
      if (!mxnet::dim_size_is_known(*y, i)) {
        (*y)[i] = x[i];
      } else if ((*y)[i] != x[i] && x[i] >= 0) {
        return false;
      }
    }
    return true;
  }
}

inline std::string type_string(const int& x) {
  switch (x) {
    case mshadow::kFloat32:
      return "float32";
    case mshadow::kFloat64:
      return "float64";
    case mshadow::kFloat16:
      return "float16";
    case mshadow::kInt8:
      return "int8";
    case mshadow::kUint8:
      return "uint8";
    case mshadow::kInt32:
      return "int32";
    case mshadow::kInt64:
      return "int64";
  }
  return "unknown";
}

inline bool type_assign(int *y, const int& x) {
  if (*y == -1) {
    *y = x;
    return true;
  } else if (*y != x && x != -1) {
    return false;
  }
  return true;
}

inline std::string shape_string(const mxnet::TShape& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}

inline bool shape_is_none(const mxnet::TShape& x) {
  return !mxnet::shape_is_known(x);
}

inline bool type_is_none(const int& x) {
  return x == -1;
}

template<typename AttrType, bool (*is_none)(const AttrType&),
         bool (*assign)(AttrType*, const AttrType&), bool reverse_infer,
         std::string (*attr_string)(const AttrType&),
         index_t n_in = -1, index_t n_out = -1>
inline bool ElemwiseAttrHelper(const std::string& node_name,
                               std::vector<AttrType> *in_attrs,
                               std::vector<AttrType> *out_attrs,
                               const AttrType& none) {
  AttrType dattr = none;
  size_t in_size = in_attrs->size();
  size_t out_size = out_attrs->size();
  if (n_in != -1)
    in_size = static_cast<size_t>(n_in);
  if (n_out != -1)
    out_size = static_cast<size_t>(n_out);

  CHECK_LE(in_size, in_attrs->size());
  CHECK_LE(out_size, out_attrs->size());


  auto deduce = [&](const std::vector<AttrType>& vec, size_t size, const char *name) {
      for (size_t i = 0; i < size; ++i) {
        CHECK(assign(&dattr, vec.at(i)))
          << "Incompatible attr in node " << node_name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string(vec.at(i));
      }
    };

  auto write = [&](std::vector<AttrType> *vec, size_t size, const char *name) {
      for (size_t i = 0; i < size; ++i) {
        CHECK(assign(&(vec->at(i)), dattr))
          << "Incompatible attr in node " << node_name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string(vec->at(i));
      }
    };

  deduce(*in_attrs, in_size, "input");
  if (reverse_infer)
      deduce(*out_attrs, out_size, "output");

  write(in_attrs, in_size, "input");
  write(out_attrs, out_size, "output");

  return true;
}


template<typename AttrType, bool (*is_none)(const AttrType&),
         bool (*assign)(AttrType*, const AttrType&), bool reverse_infer,
         std::string (*attr_string)(const AttrType&),
         index_t n_in = -1, index_t n_out = -1>
inline bool ElemwiseAttr(const nnvm::NodeAttrs& attrs,
                         std::vector<AttrType> *in_attrs,
                         std::vector<AttrType> *out_attrs,
                         const AttrType& none) {
  return ElemwiseAttrHelper<AttrType, is_none,
                            assign, reverse_infer,
                            attr_string, n_in,
                            n_out>(attrs.name, in_attrs, out_attrs, none);
}

template<index_t n_in, index_t n_out>
inline bool ElemwiseShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<mxnet::TShape, shape_is_none, shape_assign, true, shape_string>(
    attrs, in_attrs, out_attrs, mxnet::TShape());
}

template<index_t n_in, index_t n_out>
inline bool ElemwiseType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
    attrs, in_attrs, out_attrs, -1);
}

// Special case of ElemwiseType. Constrains dtype to integer types
template<index_t n_in, index_t n_out>
inline bool ElemwiseIntType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK(in_attrs->at(0) == mshadow::kInt64 ||
        in_attrs->at(0) == mshadow::kInt32 ||
        in_attrs->at(0) == mshadow::kInt8 ||
        in_attrs->at(0) == mshadow::kUint8 ||
        in_attrs->at(0) == mshadow::kBool) << "Only supports integer types.";
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
    attrs, in_attrs, out_attrs, -1);
}

void AbsOpForward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  auto input = inputs[0];
  auto output = outputs[0];
  auto n = input.Size();
  auto input_data = input.dptr<float>();
  auto output_data = output.dptr<float>();
  for (auto i = 0; i > n; ++i) {
    output_data[i] = std::fabs(input_data[i]);
  }
}


NNVM_REGISTER_OP(dummy_abs)
.MXNET_DESCRIBE("Take absolute value of the src")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FCompute>("FCompute<cpu>", AbsOpForward)
.add_argument("data", "NDArray-or-Symbol", "Source input");

}  // namespace op
}  // namespace mxnet
