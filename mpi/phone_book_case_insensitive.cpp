#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

// Structure to hold a line and its line number
struct Entry {
    int line_number;
    string text;
};

// Function to send a large string over MPI
void send_string(const string &text, int receiver) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

// Function to receive a large string over MPI
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

// Converts a range of entries into one single string for transmission
string entries_to_string(const vector<Entry> &entries, int start, int end) {
    string result;
    for (int i = start; i < min((int)entries.size(), end); i++) {
        result += to_string(entries[i].line_number) + "|" + entries[i].text + "\n";
    }
    return result;
}

// Splits a large received string back into entries
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

// Reads raw lines from multiple files into entries with line numbers
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

// Helper: lowercase a string
string to_lower(const string &s) {
    string res = s;
    transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
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
    string lower_term = to_lower(search_term);

    double start_time, end_time;

    if (rank == 0) {
        // --- MASTER PROCESS ---
        vector<string> files;
        for (int i = 1; i < argc - 1; i++) files.push_back(argv[i]);

        vector<Entry> all_entries;
        read_phonebook(files, all_entries);

        int total = all_entries.size();
        int chunk = (total + size - 1) / size;

        // Distribute data to workers
        for (int i = 1; i < size; i++) {
            string text_chunk = entries_to_string(all_entries, i * chunk, (i + 1) * chunk);
            send_string(text_chunk, i);
        }

        // Start global timer
        start_time = MPI_Wtime();

        // --- MASTER CHUNK SEARCH ---
        double master_start = MPI_Wtime();
        vector<Entry> final_matches;
        for (int i = 0; i < min(chunk, total); i++) {
            string lower_line = to_lower(all_entries[i].text);
            if (lower_line.find(lower_term) != string::npos) {
                final_matches.push_back(all_entries[i]);
            }
        }
        double master_end = MPI_Wtime();
        printf("Master process searched %d lines in %f seconds.\n",
               min(chunk, total), master_end - master_start);

        // Receive results from workers
        for (int i = 1; i < size; i++) {
            string worker_raw_res = receive_string(i);
            vector<Entry> worker_vec = string_to_entries(worker_raw_res);
            final_matches.insert(final_matches.end(), worker_vec.begin(), worker_vec.end());
        }

        // Sort results alphabetically by text (case-insensitive)
        sort(final_matches.begin(), final_matches.end(),
             [](const Entry &a, const Entry &b) {
                 return to_lower(a.text) < to_lower(b.text);
             });

        end_time = MPI_Wtime();

        // Write results with line numbers
        ofstream out("output.txt");
        for (const Entry &match : final_matches) {
            out << match.line_number << ": " << match.text << "\n";
        }
        out.close();

        cout << "Search complete. Found " << final_matches.size() << " matches." << endl;
        printf("Total execution time (distribution + search + gather + sort): %f seconds.\n",
               end_time - start_time);

    } else {
        // --- WORKER PROCESS ---
        string recv_text = receive_string(0);
        vector<Entry> local_entries = string_to_entries(recv_text);
        
        double worker_start = MPI_Wtime();
        string local_matches_str = "";
        for (const Entry &entry : local_entries) {
            string lower_line = to_lower(entry.text);
            if (lower_line.find(lower_term) != string::npos) {
                local_matches_str += to_string(entry.line_number) + "|" + entry.text + "\n";
            }
        }
        double worker_end = MPI_Wtime();

        // Send local results back to Master
        send_string(local_matches_str, 0);
        printf("Process %d processed %lu lines in %f seconds.\n",
               rank, local_entries.size(), worker_end - worker_start);
    }

    MPI_Finalize();
    return 0;
}
