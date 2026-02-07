#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

struct Entry {
    int line_number;
    string text;
};

string to_lower(const string &s) {
    string res = s;
    transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}

// Compute longest common substring between two strings (case-insensitive)
// Returns the substring itself
string longest_common_substring(const string &a, const string &b) {
    string sa = to_lower(a);
    string sb = to_lower(b);
    int n = sa.size(), m = sb.size();
    vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
    int best = 0;
    int end_pos = -1;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (sa[i-1] == sb[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
                if (dp[i][j] > best) {
                    best = dp[i][j];
                    end_pos = i-1;
                }
            }
        }
    }
    if (best > 0) {
        return sa.substr(end_pos - best + 1, best);
    }
    return "";
}

void send_string(const string &text, int receiver) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

string receive_string(int sender) {
    int len;
    MPI_Status status;
    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, &status);
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, &status);
    string res(buf);
    delete[] buf;
    return res;
}

string entries_to_string(const vector<Entry> &entries, int start, int end) {
    string result;
    for (int i = start; i < min((int)entries.size(), end); i++) {
        result += to_string(entries[i].line_number) + "|" + entries[i].text + "\n";
    }
    return result;
}

vector<Entry> string_to_entries(const string &text) {
    vector<Entry> entries;
    istringstream iss(text);
    string line;
    while (getline(iss, line)) {
        if (!line.empty()) {
            size_t pos = line.find('|');
            if (pos != string::npos) {
                int line_number = stoi(line.substr(0, pos));
                string content = line.substr(pos + 1);
                entries.push_back({line_number, content});
            }
        }
    }
    return entries;
}

void read_phonebook(const vector<string> &files, vector<Entry> &entries) {
    int line_number = 1;
    for (const string &file : files) {
        ifstream f(file);
        if (!f.is_open()) {
            cerr << "Could not open file: " << file << endl;
            continue;
        }
        string line;
        while (getline(f, line)) {
            if (!line.empty()) {
                entries.push_back({line_number, line});
            }
            line_number++;
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            cerr << "Usage: mpirun -n <procs> " << argv[0] << " <file1>... <search_term>\n";
        MPI_Finalize();
        return 1;
    }

    string search_term = argv[argc - 1];
    if (search_term == " ") {
        if (rank == 0) {
            cerr << "Invalid search term: cannot be just a space.\n";
        }
        MPI_Finalize();
        return 1;
    }

    double start_time, end_time;

    if (rank == 0) {
        vector<string> files;
        for (int i = 1; i < argc - 1; i++) files.push_back(argv[i]);

        vector<Entry> all_entries;
        read_phonebook(files, all_entries);

        int total = all_entries.size();
        int chunk = (total + size - 1) / size;

        for (int i = 1; i < size; i++) {
            string text_chunk = entries_to_string(all_entries, i * chunk, (i + 1) * chunk);
            send_string(text_chunk, i);
        }

        start_time = MPI_Wtime();

        string global_best_substring = "";
        int global_best_len = 0;

        // Master chunk
        for (int i = 0; i < min(chunk, total); i++) {
            string sub = longest_common_substring(all_entries[i].text, search_term);
            if ((int)sub.size() > global_best_len) {
                global_best_len = sub.size();
                global_best_substring = sub;
            }
        }

        // Gather from workers
        for (int i = 1; i < size; i++) {
            string worker_best = receive_string(i);
            if ((int)worker_best.size() > global_best_len) {
                global_best_len = worker_best.size();
                global_best_substring = worker_best;
            }
        }

        // Now filter all entries by global_best_substring
        vector<Entry> final_matches;
        if (global_best_len > 0) {
            for (const Entry &e : all_entries) {
                string lower_line = to_lower(e.text);
                if (lower_line.find(global_best_substring) != string::npos) {
                    final_matches.push_back(e);
                }
            }
        }

        end_time = MPI_Wtime();

        ofstream out("output.txt");
        if (!final_matches.empty()) {
            out << "Longest match substring: " << global_best_substring << "\n";
            for (const Entry &match : final_matches) {
                out << match.line_number << ": " << match.text << "\n";
            }
            cout << "Found " << final_matches.size()
                 << " contacts containing longest substring \"" << global_best_substring << "\"." << endl;
        } else {
            out << "No match found.\n";
            cout << "No match found.\n";
        }
        out.close();

        printf("Total execution time: %f seconds.\n", end_time - start_time);

    } else {
        string recv_text = receive_string(0);
        vector<Entry> local_entries = string_to_entries(recv_text);
        
        double worker_start = MPI_Wtime();
        string local_best_substring = "";
        int local_best_len = 0;
        for (const Entry &entry : local_entries) {
            string sub = longest_common_substring(entry.text, search_term);
            if ((int)sub.size() > local_best_len) {
                local_best_len = sub.size();
                local_best_substring = sub;
            }
        }
        double worker_end = MPI_Wtime();

        send_string(local_best_substring, 0);
        printf("Process %d processed %lu lines in %f seconds.\n",
               rank, local_entries.size(), worker_end - worker_start);
    }

    MPI_Finalize();
    return 0;
}
