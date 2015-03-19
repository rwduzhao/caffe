/* rwduzhao */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;

std::vector<std::pair<std::string, std::pair<int, std::string> > > read_data_info(const std::string data_info_file) {
  std::ifstream data_info_stream(data_info_file.c_str());

  std::vector<std::pair<std::string, std::pair<int, std::string> > > data_infos;
  std::string name;
  std::string path;
  int label;

  while (data_info_stream >> name >> label >> path)
    data_infos.push_back(std::make_pair(name, std::make_pair(label, path)));

  return data_infos;
}

int get_feature_dimension(std::string data_file) {
  std::ifstream data_stream(data_file.c_str());
  std::string data_line;
  std::getline(data_stream, data_line);
  std::stringstream string_stream(data_line);
  int n_dim = 0;
  float x = 0.0;
  while (true) {
    string_stream >> x;
    if (!string_stream)
      break;
    ++n_dim;
  }

  return n_dim;
}

bool read_text_to_datum(const std::string data_file, const int label,
                       const int channel, const int height, const int width, Datum * datum) {
  datum->set_channels(channel);
  datum->set_height(height);
  datum->set_width(width);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();

  std::ifstream data_stream(data_file.c_str());
  std::string data_line;
  float x = 0.0;
  while (std::getline(data_stream, data_line)) {
    std::stringstream string_stream(data_line);
    int n_dim = 0;
    while (true) {
      string_stream >> x;
      if (!string_stream)
        break;
      datum_string->push_back(static_cast<char>(x));
      ++n_dim;
    }
    if (width != n_dim)
      LOG(ERROR) << "mismatching dimension in " << data_file;
  }

  return true;
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("convert_text_features info_file db_path\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  /* info file */
  std::string data_info_file = argv[1];
  std::vector<std::pair<std::string, std::pair<int, std::string> > > data_infos = read_data_info(data_info_file);
  LOG(INFO) << "total number of examples: " << data_infos.size();
  int n_dim = get_feature_dimension(data_infos[0].second.second);
  LOG(INFO) << "feature dimension: " << n_dim;

  /* db */
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;
  std::string db_path = argv[2];
  LOG(INFO) << "lmdb " << db_path << " opening";
  CHECK_EQ(mkdir(db_path.c_str(), 0744), 0) << "mkdir " << db_path << "failed";
  CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS) << "mdb_env_set_mapsize failed";  /* 1TB */
  CHECK_EQ(mdb_env_open(mdb_env, db_path.c_str(), 0, 0664), MDB_SUCCESS) << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS) << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS) << "mdb_open failed. does the lmdb already exist?";

  const int channel = atoi(argv[3]);
  const int height = atoi(argv[4]);

  /* write batches */
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  for (int data_id = 0; data_id < data_infos.size(); ++data_id) {
    std::string data_name = data_infos[data_id].first;
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", data_id, data_name.c_str());
    string keystr(key_cstr);

    int data_label = data_infos[data_id].second.first;
    std::string data_path = data_infos[data_id].second.second;
    const int width = n_dim;
    if (!read_text_to_datum(data_path, data_label, channel, height, width, &datum))
      continue;
    std::string value;
    datum.SerializeToString(&value);

    /* put in db */
    mdb_key.mv_size = keystr.size();
    mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
    mdb_data.mv_size = value.size();
    mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
    CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS) << "mdb_put failed";

    if (++count % 1000 == 0) {
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
      CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS) << "mdb_txn_begin failed";
      LOG(INFO) << "lmdb " << db_path << " processed " << count << " files";
    }
  }
  /* write the last batch */
  if (count % 1000 != 0) {
    CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
    LOG(INFO) << "lmdb " << db_path << " processed " << count << " files";
  }
  mdb_close(mdb_env, mdb_dbi);
  mdb_env_close(mdb_env);
  LOG(INFO) << "lmdb " << db_path << " closed";

  return 0;
}
